#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.
    For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.
    Returns:
        the centered image.
    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
           than three or if the number of channels in `image` doesn't match the
           number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def build_test_time_data_augmentation(x):
    """
    test time data augmentation
    input batch, output batch * 8
    x = [batch, height, width, channel]
    """
    x_rot_90 = tf.contrib.image.rotate(x, 90)
    x_rot_180 = tf.contrib.image.rotate(x, 180)
    x_rot_270 = tf.contrib.image.rotate(x, 270)

    x_flip = tf.reverse(x, [2])
    x_flip_rot_90 = tf.contrib.image.rotate(x_flip, 90)
    x_flip_rot_180 = tf.contrib.image.rotate(x_flip, 180)
    x_flip_rot_270 = tf.contrib.image.rotate(x_flip, 270)

    x = tf.concat(
        [
            x, x_rot_90, x_rot_180, x_rot_270, x_flip, x_flip_rot_90,
            x_flip_rot_180, x_flip_rot_270
        ],
        axis=0)

    return x


def build_test_time_vote(logits):
    """

    """
    logits = tf.one_hot(tf.argmax(logits, axis=1), depth=logits.shape[1])

    [
        logits, logits_rot_90, logits_rot_180, logits_rot_270, logits_flip,
        logits_flip_rot_90, logits_flip_rot_180, logits_flip_rot_270
    ] = tf.split(logits, 8)

    logits = logits + logits_rot_90 + logits_rot_180 + logits_rot_270 + logits_flip + logits_flip_rot_90 + logits_flip_rot_180 + logits_flip_rot_270

    return logits

