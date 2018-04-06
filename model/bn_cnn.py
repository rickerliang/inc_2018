#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.deformable_conv2d import build_deformable_conv2d_layer, build_deformable_conv2d_layer_2


def build_bn_cnn_4(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_4'

    with tf.variable_scope(scope_name):

        to_next_layer = build_cnn_bn_pool_layer(image_batch, training, 1)[0]

        to_next_layer = build_cnn_bn_pool_layer(to_next_layer, training, 2)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=64)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=64)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_6(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_6'

    with tf.variable_scope(scope_name):

        # 128 x 128 x 32
        to_next_layer = build_cnn_bn_pool_layer(image_batch, training, 1)[0]

        # 64 x 64 x 32
        to_next_layer = build_cnn_bn_pool_layer(to_next_layer, training, 2)[0]

        # 32 x 32 x 64
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=64)[0]

        # 16 x 16 x 64
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=64)[0]

        # 8 x 8 x 128
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=128)[0]

        # 4 x 4 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=256, bn_momentum=0.9)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_8(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_8'

    with tf.variable_scope(scope_name):

        # 128 x 128 x 64
        to_next_layer = build_cnn_bn_pool_layer(
            image_batch, training, 1, num_filter=64)[0]

        # 64 x 64 x 64
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 2, num_filter=64)[0]

        # 32 x 32 x 128
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=128)[0]

        # 16 x 16 x 128
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=128)[0]

        # 8 x 8 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=256)[0]

        # 4 x 4 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=256, bn_momentum=0.9)[0]

        # 2 x 2 x 512
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 7, num_filter=512, bn_momentum=0.9)[0]

        # 1 x 1 x 1024
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 8, num_filter=1024, bn_momentum=0.9)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_12(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_12'

    with tf.variable_scope(scope_name):

        # 128 x 128 x 64
        to_next_layer = build_cnn_bn_layer(
            image_batch, training, 1, num_filter=64)[0]
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 2, num_filter=64)[0]

        # 64 x 64 x 64
        to_next_layer = build_cnn_bn_layer(
            to_next_layer, training, 3, num_filter=64)[0]
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=64)[0]

        # 32 x 32 x 128
        to_next_layer = build_cnn_bn_layer(
            to_next_layer, training, 5, num_filter=128)[0]
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=128)[0]

        # 16 x 16 x 128
        to_next_layer = build_cnn_bn_layer(
            to_next_layer, training, 7, num_filter=128)[0]
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 8, num_filter=128)[0]

        # 8 x 8 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 9, num_filter=256)[0]

        # 4 x 4 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 10, num_filter=256, bn_momentum=0.9)[0]

        # 2 x 2 x 512
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 11, num_filter=512, bn_momentum=0.9)[0]

        # 1 x 1 x 1024
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 12, num_filter=1024, bn_momentum=0.9)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_8_crelu_deformable(image_batch, training):
    """

        :param image_batch:
        :return:
        """
    scope_name = 'bn_cnn_8_crelu_deformable'

    with tf.variable_scope(scope_name):
        # 128 x 128 x 64 (crelu)
        to_next_layer = build_cnn_bn_pool_layer(
            image_batch, training, 1, num_filter=32,
            activation_fn=tf.nn.crelu)[0]

        # 64 x 64 x 64 (crelu)
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer,
            training,
            2,
            num_filter=32,
            activation_fn=tf.nn.crelu)[0]

        # 32 x 32 x 128
        #to_next_layer = build_cnn_bn_pool_layer(
        #    to_next_layer, training, 3, num_filter=128)[0]
        to_next_layer = build_deformable_conv2d_layer_2(
            to_next_layer, training, 3, num_filter=128)

        # 16 x 16 x 128
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=128)[0]

        # 8 x 8 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=256)[0]

        # 4 x 4 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=256)[0]

        # 2 x 2 x 512
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 7, num_filter=512)[0]

        # 1 x 1 x 1024
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 8, num_filter=1024)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_8_crelu_with_dropout(image_batch, training):
    """
    drops the lower feature maps
    """
    scope_name = 'bn_cnn_8_crelu_with_dropout'

    with tf.variable_scope(scope_name):

        to_next_layer = tf.layers.dropout(
            image_batch,
            0.1,
            noise_shape=[
                tf.shape(image_batch)[0], 1, 1,
                tf.shape(image_batch)[3]
            ],
            training=training,
            name='image_drop')

        # 128 x 128 x 64 (crelu)
        to_next_layer = build_cnn_bn_pool_layer(
            image_batch, training, 1, num_filter=32,
            activation_fn=tf.nn.crelu)[0]

        to_next_layer = tf.layers.dropout(
            to_next_layer,
            0.1,
            noise_shape=[
                tf.shape(to_next_layer)[0], 1, 1,
                tf.shape(to_next_layer)[3]
            ],
            training=training,
            name='cnn_1_drop')

        # 64 x 64 x 64 (crelu)
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer,
            training,
            2,
            num_filter=32,
            activation_fn=tf.nn.crelu)[0]

        to_next_layer = tf.layers.dropout(
            to_next_layer,
            0.1,
            noise_shape=[
                tf.shape(to_next_layer)[0], 1, 1,
                tf.shape(to_next_layer)[3]
            ],
            training=training,
            name='cnn_2_drop')

        # 32 x 32 x 128
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=128)[0]

        # 16 x 16 x 128
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=128)[0]

        # 8 x 8 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=256)[0]

        # 4 x 4 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=256)[0]

        # 2 x 2 x 512
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 7, num_filter=512)[0]

        # 1 x 1 x 1024
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 8, num_filter=1024)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_8_crelu(image_batch, training):
    """

        :param image_batch:
        :return:
        """
    scope_name = 'bn_cnn_8_crelu'

    with tf.variable_scope(scope_name):
        to_next_layer = build_cnn_bn_pool_layer(
            image_batch, training, 1, num_filter=16,
            activation_fn=tf.nn.crelu)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer,
            training,
            2,
            num_filter=16,
            activation_fn=tf.nn.crelu)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=64)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=64)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=128)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=128)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 7, num_filter=256)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 8, num_filter=256)[0]

        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 9, num_filter=512)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_8_crelu_2(image_batch, training):
    """

        :param image_batch:
        :return:
        """
    scope_name = 'bn_cnn_8_crelu'

    with tf.variable_scope(scope_name):
        # 128 x 128 x 128 (crelu)
        to_next_layer = build_cnn_bn_pool_layer(
            image_batch, training, 1, num_filter=64,
            activation_fn=tf.nn.crelu)[0]

        # 64 x 64 x 128 (crelu)
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer,
            training,
            2,
            num_filter=64,
            activation_fn=tf.nn.crelu)[0]

        # 32 x 32 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=256)[0]

        # 16 x 16 x 256
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 4, num_filter=256)[0]

        # 8 x 8 x 512
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=512)[0]

        # 4 x 4 x 512
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=512)[0]

        # 2 x 2 x 1024
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 7, num_filter=1024)[0]

        # 1 x 1 x 2048
        to_next_layer = build_cnn_bn_pool_layer(
            to_next_layer, training, 8, num_filter=2048)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_bn_cnn_6_with_skip_connection(image_batch, training):
    """

    :param image_batch:
    :return:
    """
    scope_name = 'bn_cnn_6_with_skip_connection'

    with tf.variable_scope(scope_name):

        # 128 x 128 x 32
        to_next_layer = build_cnn_bn_pool_layer(image_batch, training, 1)[0]

        # 64 x 64 x 32
        to_next_layer = build_cnn_bn_pool_layer(to_next_layer, training, 2)[0]

        # 32 x 32 x 64
        cnn_3 = build_cnn_bn_pool_layer(
            to_next_layer, training, 3, num_filter=64)[0]

        # 16 x 16 x 64
        to_next_layer, before_pooling_4, _, _ = build_cnn_bn_pool_layer(
            cnn_3, training, 4, num_filter=64, swap_pooling_pos=False)

        # 8 x 8 x 128
        to_next_layer, before_pooling_5, _, _ = build_cnn_bn_pool_layer(
            to_next_layer, training, 5, num_filter=128, swap_pooling_pos=False)

        # 4 x 4 x 256
        _, before_pooling_6, _, _ = build_cnn_bn_pool_layer(
            to_next_layer, training, 6, num_filter=256, swap_pooling_pos=False)

        resized_5 = tf.image.resize_images(
            before_pooling_5, [cnn_3.shape[1], cnn_3.shape[2]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        resized_6 = tf.image.resize_images(
            before_pooling_6, [cnn_3.shape[1], cnn_3.shape[2]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        feature = tf.concat(
            [cnn_3, before_pooling_4, resized_5, resized_6],
            axis=3,
            name='skip_concat')

        to_next_layer = build_cnn_bn_pool_layer(feature, training, 7)[0]

        flatten = tf.layers.flatten(to_next_layer, name='flatten')

    return flatten


def build_cnn_bn_layer(image_batch,
                       training,
                       layer_number,
                       num_filter=32,
                       kernel_size=2,
                       strides=(1, 1),
                       bn_momentum=0.9,
                       activation_fn=tf.nn.leaky_relu):
    conv_name = 'conv_{0}'.format(layer_number)
    bn_name = 'bn_{0}'.format(layer_number)
    activation_name = 'activation_{0}'.format(layer_number)

    cnn_out = tf.layers.conv2d(
        inputs=image_batch,
        filters=num_filter,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        name=conv_name)

    bn_out = tf.layers.batch_normalization(
        cnn_out, momentum=bn_momentum, training=training, name=bn_name)

    activation_out = activation_fn(bn_out, name=activation_name)
    return activation_out, bn_out, cnn_out


def build_cnn_bn_pool_layer(image_batch,
                            training,
                            layer_number,
                            num_filter=32,
                            kernel_size=2,
                            strides=(1, 1),
                            bn_momentum=0.9,
                            pool_size=2,
                            pool_strides=2,
                            swap_pooling_pos=True,
                            activation_fn=tf.nn.leaky_relu,
                            conv_padding="same",
                            pool_padding="same"):
    """
    cnn -> bn -> max_pooling -> leaky_relu
    note:
      this layer dose not implemented as common way:(cnn -> bn -> activation -> pooling)
      because exchange the positions of pooling and activation can reduce computation costs (iif pooling is max_pooling)
    :param image_batch:
    :param training:
    :param layer_number:
    :param num_filter:
    :param kernel_size:
    :param strides:
    :param bn_momentum:
    :param pool_size:
    :param pool_strides:
    :return:
    """

    conv_name = 'conv_{0}'.format(layer_number)
    bn_name = 'bn_{0}'.format(layer_number)
    activation_name = 'activation_{0}'.format(layer_number)
    pool_name = 'pool_{0}'.format(layer_number)

    cnn_out = tf.layers.conv2d(
        inputs=image_batch,
        filters=num_filter,
        kernel_size=kernel_size,
        strides=strides,
        padding=conv_padding,
        use_bias=False,
        name=conv_name)

    bn_out = tf.layers.batch_normalization(
        cnn_out, momentum=bn_momentum, training=training, name=bn_name)

    if swap_pooling_pos:
        pooling_out = tf.layers.max_pooling2d(
            inputs=bn_out,
            pool_size=pool_size,
            strides=pool_strides,
            padding=pool_padding,
            name=pool_name)

        activation_out = activation_fn(pooling_out, name=activation_name)

        return activation_out, pooling_out, bn_out, cnn_out
    else:
        activation_out = activation_fn(bn_out, name=activation_name)

        pooling_out = tf.layers.max_pooling2d(
            inputs=activation_out,
            pool_size=pool_size,
            strides=pool_strides,
            padding=pool_padding,
            name=pool_name)

        return pooling_out, activation_out, bn_out, cnn_out
