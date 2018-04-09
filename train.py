#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.framework import ops as _ops
import horovod.tensorflow as hvd

from model.model_class import InceptionResnetV2, Resnet50V2, Cnn8CreluLsoftmax, MobileNetV2
from gen_data.data_provider import get_tf_dataset


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size_per_replica", help="", default=80,
                        type=int)
    parser.add_argument("--num_epoch", help="", default=999, type=int)
    parser.add_argument("--early_stopping_step", help="", default=1, type=int)
    parser.add_argument("--lambda_decay_init", help="", default=1000.0,
                        type=float)
    parser.add_argument("--lambda_decay_steps", help="", default=4000, type=int)
    parser.add_argument("--lambda_decay_rate", help="", default=0.8, type=float)
    parser.add_argument("--lambda_decay_min", help="", default=9.0, type=float)
    parser.add_argument("--tfdbg", help="", default=False, type=bool)
    parser.add_argument("--use_horovod", default=False, type=bool)
    args = parser.parse_args()
    print(args)
    return args


class Supervisor:

    def __init__(self):
        print("parse argument...")
        self.args = parse_argument()

        print("horovod stuff...")
        self.number_replica, self.rank = self.horovod_stuff(self.args)

        print("build input pipeline...")
        train_dataset_filename = "_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_val__7_dataset.txt"

        self.image_size = 384
        dataset, self.number_class, number_examples = get_tf_dataset(
            train_dataset_filename,
            balance_count=20,
            parallel_calls=64,
            batch_size=self.args.batch_size_per_replica,
            crop_size=self.image_size,
            prefetch_count=self.args.batch_size_per_replica * 2,
            num_shards=self.number_replica,
            index=self.rank)
        dataset = dataset.repeat()
        self.number_examples = int(number_examples / self.number_replica)
        iterator = dataset.make_one_shot_iterator()
        self.train_images, self.train_labels = iterator.get_next()

        val_dataset_filename = "_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_val__7_dataset.txt"
        val_dataset, _, self.val_number_examples = get_tf_dataset(
            val_dataset_filename,
            balance_count=0,
            parallel_calls=64,
            batch_size=self.args.batch_size_per_replica,
            crop_size=self.image_size,
            validation=True
        )
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_one_shot_iterator()
        self.val_images, self.val_labels = val_iterator.get_next()

        self.x = tf.placeholder(
            tf.float32,
            shape=(None, self.image_size, self.image_size, 3),
            name='x')
        self.y = tf.placeholder(tf.int64, shape=(None), name='y')
        self.is_training = tf.placeholder(tf.bool, name='phase')

        if self.args.tfdbg:
            self.image_summary()

        self.global_step = tf.train.create_global_step()

        lambda_decay = tf.train.exponential_decay(
            self.args.lambda_decay_init,
            self.global_step,
            self.args.lambda_decay_steps,
            self.args.lambda_decay_rate,
            staircase=True,
            name='lambda_decay')

        lambda_decay = tf.cond(lambda_decay > self.args.lambda_decay_min,
                               lambda: lambda_decay,
                               lambda: tf.constant(self.args.lambda_decay_min))

        tf.summary.scalar("lambda_decay", lambda_decay)

        print("build model...")
        self.model = Cnn8CreluLsoftmax(self.args.use_horovod)
        self.train_images = self.model.preprocessing(self.train_images)
        self.val_images = self.model.preprocessing(self.val_images)
        linear, logits, trainable_var = self.model.build_model(self.x, self.y,
                                                          self.number_class,
                                                          lambda_decay,
                                                          self.is_training)

        print("build loss...")
        self.loss_op = self.model.build_loss(self.y, linear)
        tf.summary.scalar("total_loss", self.loss_op)

        print("build train op...")
        self.train_op = self.model.build_train_op(self.loss_op, self.global_step,
                                             trainable_var)

        print("build accuracy op...")
        correct_prediction = tf.equal(tf.squeeze(self.y), tf.argmax(linear, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('batch_accuracy', self.accuracy_op)

        self.session_config = tf.ConfigProto()
        self.session_config.gpu_options.allow_growth = True
        if self.args.use_horovod:
            self.session_config.gpu_options.visible_device_list = str(
                hvd.local_rank())

    def image_summary(self):
        for i in range(8):
            log_image = gen_logging_ops._image_summary(tf.as_string(self.train_labels[i]),
                                       tf.expand_dims(self.train_images[i], 0),
                                       max_images=1)
            _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)

    def horovod_stuff(self, args):
        number_replica = 1
        rank = 0
        if args.use_horovod:
            hvd.init()
            number_replica = hvd.size()
            rank = hvd.rank()
            print("use horovod, number replica:{0}, rank:{1}".format(number_replica, rank))
        else:
            print("do not use horovod")

        return number_replica, rank


    def train(self):

        with tf.Session(config=self.session_config) as session:
            print("session run...")
            if self.args.tfdbg:
                session = tf_debug.LocalCLIDebugWrapperSession(session)

            session.run(tf.global_variables_initializer())
            self.model.load_pretrained_weight(session)
            if self.args.use_horovod:
                session.run(hvd.broadcast_global_variables(0))

            self.training_process(session)

            print("train complete")


    def training_process(self, session):

        best_validation_accuracy = 0

        merge_summary = tf.summary.merge_all()

        model_saver = tf.train.Saver(max_to_keep=15)
        save_path = 'model_weight_{0}'.format(self.rank)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        summary_writer = tf.summary.FileWriter(save_path, session.graph)

        best_model_save_path = 'model_best_weight'
        if not os.path.exists(best_model_save_path):
            os.makedirs(best_model_save_path)
        best_model_saver = tf.train.Saver(max_to_keep=10)

        early_stop_step = 0
        for i in range(self.args.num_epoch):
            loss_avg = self.training_phase(i, merge_summary, session,
                                           summary_writer)

            model_saver.save(
                session,
                os.path.join(
                    save_path,
                    "inat_2018_{0:.8f}.ckpt".format(loss_avg)),
                global_step=self.global_step)

            if (self.args.use_horovod and self.rank == 0) or self.args.use_horovod is False:
                validation_acc = self.validation_phase(session)
                if validation_acc > best_validation_accuracy:
                    best_validation_accuracy = validation_acc
                    early_stop_step = 0
                    print(
                        '========================= save best model ======================='
                    )
                    best_model_saver.save(
                        session,
                        os.path.join(
                            best_model_save_path,
                            "inat_2018_classifier_{0:.4f}_{1:.8f}.ckpt".format(
                                best_validation_accuracy, loss_avg)),
                        global_step=self.global_step)
                else:
                    early_stop_step += 1

                if early_stop_step >= self.args.early_stopping_step:
                    print("early stop...")
                    return


    def training_phase(self, epoch, merge_summary, session, summary_writer):
        accuracy_avg = 0.0
        loss_avg = 0.0
        for j in range(int(math.ceil(self.number_examples / self.args.batch_size_per_replica))):
            images, labels = session.run([self.train_images, self.train_labels])
            if j == 0:
                step, summary, loss_value, accuracy_value, _ = session.run(
                    [self.global_step, merge_summary, self.loss_op,
                     self.accuracy_op, self.train_op],
                    feed_dict={self.x: images,
                               self.y: labels,
                               self.is_training: True})
                summary_writer.add_summary(summary, step)
            else:
                loss_value, accuracy_value, _ = session.run(
                    [self.loss_op, self.accuracy_op, self.train_op],
                    feed_dict={self.x: images,
                               self.y: labels,
                               self.is_training: True})

            accuracy_avg = accuracy_avg + (accuracy_value - accuracy_avg) / (j + 1)
            loss_avg = loss_avg + (loss_value - loss_avg) / (j + 1)
            sys.stdout.write("\r{0}--{1} training avg loss:{2} batch loss:{3}    ".
                             format(epoch, j, loss_avg, loss_value))
            sys.stdout.flush()
        print("")
        print("training acc:{0}".format(accuracy_avg))
        return loss_avg

    def validation_phase(self, session):
        accuracy_avg = 0.0
        loss_avg = 0.0
        for j in range(int(math.ceil(self.val_number_examples / self.args.batch_size_per_replica))):
            images, labels = session.run([self.val_images, self.val_labels])
            loss_value, accuracy_value = session.run(
                [self.loss_op, self.accuracy_op],
                feed_dict={self.x: images,
                           self.y: labels,
                           self.is_training: False})

            accuracy_avg = accuracy_avg + (accuracy_value - accuracy_avg) / (j + 1)
            loss_avg = loss_avg + (loss_value - loss_avg) / (j + 1)
            sys.stdout.write("\r{0} validation avg loss:{1} batch loss:{2}    ".
                             format(j, loss_avg, loss_value))
            sys.stdout.flush()
        print("")
        print("validation acc:{0}".format(accuracy_avg))
        return accuracy_avg


supervisor = Supervisor()
supervisor.train()
