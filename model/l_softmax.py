#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.special import binom


def l_softmax(input, target, num_class, margin, lambda_decay, training, name):
    """
    https://arxiv.org/abs/1612.02295
    https://github.com/jihunchoi/lsoftmax-pytorch/blob/master/lsoftmax.py
    :return:
    """

    with tf.variable_scope(name):
        weight = tf.get_variable(
            'weight',
            shape=[input.shape[1], num_class],
            initializer=tf.contrib.layers.xavier_initializer())

        logits = tf.cond(
            training,
            lambda: l_softmax_training(input, target, margin, lambda_decay, weight),
            lambda: tf.matmul(input, weight))
        # logits = l_softmax_training(input, target, margin, weight)
        return logits


def find_k(cos, divisor):
    # stop gradient
    acos = tf.acos(cos)
    k = tf.floor(acos / divisor)
    return tf.stop_gradient(k)


def l_softmax_training(input, target, margin, lambda_decay, weight):

    divisor = tf.constant(np.pi / margin)
    coeffs = tf.constant(binom(margin, range(0, margin + 1, 2)), tf.float32)
    cos_exps_range = [i for i in range(margin, -1, -2)]
    cos_exps = tf.constant(cos_exps_range, tf.float32)
    sin_sq_exps_range = [i for i in range(len(cos_exps_range))]
    sin_sq_exps = tf.constant(sin_sq_exps_range, tf.float32)
    signs = [1]
    for i in range(1, len(sin_sq_exps_range)):
        signs.append(signs[-1] * -1)
    signs = tf.constant(signs, tf.float32)

    # weight = tf.nn.l2_normalize(weight, dim=1, name='weight_l2_normalize')
    input_norm = tf.norm(
        input, ord=2, axis=1, keepdims=True, name='input_l2_norm')
    weight_norm = tf.norm(
        weight, ord=2, axis=0, keepdims=True, name='weight_l2_norm')

    logits = tf.matmul(input, weight)

    batch_index = tf.cast(tf.range(tf.shape(input)[0]), tf.int64)
    # [[sample_0, sample_0_target_index], [sample_1, sample_1_target_index], [sample_2, sample_2_target_index]]
    logits_target_indices = tf.cast(
        tf.transpose(tf.stack([batch_index, target])), tf.int32)

    logits_target = tf.gather_nd(logits, logits_target_indices)
    weight_target_norm = tf.gather(weight_norm, target, axis=1)

    norm_target_prod = tf.transpose(weight_target_norm) * input_norm
    # cos_target: (batch_size,)
    # Multiple-Angle Formulas
    cos_target = tf.expand_dims(
        logits_target, axis=1) / (norm_target_prod + 1e-10)
    sin_sq_target = 1 - cos_target**2

    cos_terms = tf.pow(cos_target, tf.expand_dims(cos_exps, 0))
    sin_sq_terms = tf.pow(sin_sq_target, tf.expand_dims(sin_sq_exps, 0))
    cosm_terms = (tf.expand_dims(signs, 0) * tf.expand_dims(coeffs, 0) *
                  cos_terms * sin_sq_terms)
    cosm = tf.reduce_sum(cosm_terms, 1, keepdims=True)

    k = find_k(cos_target, divisor)
    ls_target = norm_target_prod * ((tf.pow(-1., k) * cosm) - 2. * k)
    # f_yi = (lambda*|W_yi|*|x_i|*cos(theta_yi) + |W_yi|*|x_i|*psi(theta_yi)) / (1+lambda)
    ls_target = (
        lambda_decay * tf.expand_dims(logits_target, axis=1) + ls_target) / (
            1. + lambda_decay)
    """
    updated = [[0, x, 0, 0],
               [0, 0, x, 0],
               [0, x, 0, 0],
               [0, 0, 0, x]]
               
    mask = [[False, True, False, False],
            [False, False, True, False],
            [False, True, False, False],
            [False, False, False, True]]
    
    # finally logits are        
    logits = [[logit, x, logit, logit],
              [logit, logit, x, logit],
              [logit, x, logit, logit],
              [logit, logit, logit, x]]
    """
    updated = tf.scatter_nd(logits_target_indices,
                            tf.squeeze(ls_target), tf.shape(logits))
    mask = tf.cast(
        tf.sparse_to_dense(logits_target_indices, tf.shape(logits), 1), tf.bool)
    logits = tf.where(mask, updated, logits)

    return logits


def a_softmax():
    """
    https://arxiv.org/abs/1704.08063
    :return:
    """
    return None
