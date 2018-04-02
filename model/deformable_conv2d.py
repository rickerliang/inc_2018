#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def build_deformable_conv2d_layer(input,
                                  training,
                                  layer_number,
                                  num_filter=32,
                                  kernel_size=2,
                                  bn_momentum=0.9,
                                  pool_size=2,
                                  pool_strides=2,
                                  activation_fn=tf.nn.leaky_relu):
    layer_name = "deformable_conv2d_{0}".format(layer_number)
    pool_name = 'pool_{0}'.format(layer_number)
    bn_name = 'bn_{0}'.format(layer_number)
    activation_name = 'activation_{0}'.format(layer_number)

    cnn_out = DeformConv2D(input, kernel_size, num_filter, layer_name, 1,
                           True).infer()

    bn_out = tf.layers.batch_normalization(
        cnn_out, momentum=bn_momentum, training=training, name=bn_name)

    pooling_out = tf.layers.max_pooling2d(
        inputs=bn_out,
        pool_size=pool_size,
        strides=pool_strides,
        padding='same',
        name=pool_name)

    activation_out = activation_fn(pooling_out, name=activation_name)

    return activation_out


def build_deformable_conv2d_layer_2(input,
                                    training,
                                    layer_number,
                                    num_filter=32,
                                    kernel_size=2,
                                    bn_momentum=0.9,
                                    pool_size=2,
                                    pool_strides=2,
                                    activation_fn=tf.nn.leaky_relu):

    layer_name = "deformable_conv2d_{0}".format(layer_number)
    pool_name = 'pool_{0}'.format(layer_number)
    bn_name = 'bn_{0}'.format(layer_number)
    activation_name = 'activation_{0}'.format(layer_number)

    with tf.variable_scope("deformable_conv2d_layer_{0}".format(layer_number)):

        batch, i_h, i_w, i_c = input.get_shape().as_list()
        cnn_out = deform_conv2d(
            input,
            offset_shape=[i_h, i_w, i_c, 2 * kernel_size * kernel_size],
            filter_shape=[kernel_size, kernel_size, i_c, num_filter],
            activation=None,
            scope=layer_name)

        bn_out = tf.layers.batch_normalization(
            cnn_out, momentum=bn_momentum, training=training, name=bn_name)

        pooling_out = tf.layers.max_pooling2d(
            inputs=bn_out,
            pool_size=pool_size,
            strides=pool_strides,
            padding='same',
            name=pool_name)

        activation_out = activation_fn(pooling_out, name=activation_name)

    return activation_out


"""
https://github.com/maestrojeong/deformable_convnet/blob/master/ops.py
"""


def conv2d(input_,
           filter_shape,
           strides=[1, 1, 1, 1],
           padding=False,
           activation=None,
           batch_norm=False,
           istrain=False,
           scope=None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - activation function
            default to be None
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else:
            padding = 'VALID'
        w = tf.get_variable(
            name="w",
            shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(
                uniform=False))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(
                conv,
                center=True,
                scale=True,
                decay=0.8,
                is_training=istrain,
                scope='batch_norm')
            if activation is None:
                return norm
            return activation(norm)
        else:
            b = tf.get_variable(
                name="b",
                shape=filter_shape[-1],
                initializer=tf.constant_initializer(0.001))
            if activation is None:
                return conv + b
            return activation(conv + b)


def deform_conv2d(x, offset_shape, filter_shape, activation=None, scope=None):
    '''
    Args:
        x - 4D tensor [batch, i_h, i_w, i_c] NHWC format
        offset_shape - list with 4 elements
            [o_h, o_w, o_ic, o_oc]
        filter_shape - list with 4 elements
            [f_h, f_w, f_ic, f_oc]
    '''

    batch, i_h, i_w, i_c = x.get_shape().as_list()
    f_h, f_w, f_ic, f_oc = filter_shape
    o_h, o_w, o_ic, o_oc = offset_shape
    assert f_ic == i_c and o_ic == i_c, "# of input_channel should match but %d, %d, %d" % (
        i_c, f_ic, o_ic)
    assert o_oc == 2 * f_h * f_w, "# of output channel in offset_shape should be 2*filter_height*filter_width but %d and %d" % (
        o_oc, 2 * f_h * f_w)

    with tf.variable_scope(scope or "deform_conv"):
        offset_map = conv2d(
            x, offset_shape, padding=True, scope="offset_conv"
        )  # offset_map : [batch, i_h, i_w, #(=2*f_h*f_w)]
    offset_map = tf.reshape(offset_map, [batch, i_h, i_w, f_h, f_w, 2])
    offset_map_h = tf.tile(
        tf.reshape(offset_map[..., 0], [batch, i_h, i_w, f_h, f_w]),
        [i_c, 1, 1, 1, 1])  # offset_map_h [batch*i_c, i_h, i_w, f_h, f_w]
    offset_map_w = tf.tile(
        tf.reshape(offset_map[..., 1], [batch, i_h, i_w, f_h, f_w]),
        [i_c, 1, 1, 1, 1])  # offset_map_w [batch*i_c, i_h, i_w, f_h, f_w]

    coord_w, coord_h = tf.meshgrid(
        tf.range(i_w, dtype=tf.float32), tf.range(
            i_h,
            dtype=tf.float32))  # coord_w : [i_h, i_w], coord_h : [i_h, i_w]
    coord_fw, coord_fh = tf.meshgrid(
        tf.range(f_w, dtype=tf.float32), tf.range(
            f_h,
            dtype=tf.float32))  # coord_fw : [f_h, f_w], coord_fh : [f_h, f_w]
    '''
    coord_w 
        [[0,1,2,...,i_w-1],...]
    coord_h
        [[0,...,0],...,[i_h-1,...,i_h-1]]
    '''
    coord_h = tf.tile(
        tf.reshape(coord_h, [1, i_h, i_w, 1, 1]),
        [batch * i_c, 1, 1, f_h,
         f_w])  # coords_h [batch*i_c, i_h, i_w, f_h, f_w)
    coord_w = tf.tile(
        tf.reshape(coord_w, [1, i_h, i_w, 1, 1]),
        [batch * i_c, 1, 1, f_h,
         f_w])  # coords_w [batch*i_c, i_h, i_w, f_h, f_w)

    coord_fh = tf.tile(
        tf.reshape(coord_fh, [1, 1, 1, f_h, f_w]),
        [batch * i_c, i_h, i_w, 1,
         1])  # coords_fh [batch*i_c, i_h, i_w, f_h, f_w)
    coord_fw = tf.tile(
        tf.reshape(coord_fw, [1, 1, 1, f_h, f_w]),
        [batch * i_c, i_h, i_w, 1,
         1])  # coords_fw [batch*i_c, i_h, i_w, f_h, f_w)

    coord_h = coord_h + coord_fh + offset_map_h
    coord_w = coord_w + coord_fw + offset_map_w
    coord_h = tf.clip_by_value(
        coord_h, clip_value_min=0,
        clip_value_max=i_h - 1)  # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_w = tf.clip_by_value(
        coord_w, clip_value_min=0,
        clip_value_max=i_w - 1)  # [batch*i_c, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(tf.floor(coord_h),
                       tf.int32)  # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_hM = tf.cast(tf.ceil(coord_h),
                       tf.int32)  # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wm = tf.cast(tf.floor(coord_w),
                       tf.int32)  # [batch*i_c, i_h, i_w, f_h, f_w]
    coord_wM = tf.cast(tf.ceil(coord_w),
                       tf.int32)  # [batch*i_c, i_h, i_w, f_h, f_w]

    x_r = tf.reshape(tf.transpose(x, [3, 0, 1, 2]),
                     [-1, i_h, i_w])  # [i_c*batch, i_h, i_w]

    bc_index = tf.tile(
        tf.reshape(tf.range(batch * i_c), [-1, 1, 1, 1, 1]),
        [1, i_h, i_w, f_h, f_w])

    coord_hmwm = tf.concat(
        values=[
            tf.expand_dims(bc_index, -1),
            tf.expand_dims(coord_hm, -1),
            tf.expand_dims(coord_wm, -1)
        ],
        axis=-1
    )  # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wm)
    coord_hmwM = tf.concat(
        values=[
            tf.expand_dims(bc_index, -1),
            tf.expand_dims(coord_hm, -1),
            tf.expand_dims(coord_wM, -1)
        ],
        axis=-1
    )  # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hm, coord_wM)
    coord_hMwm = tf.concat(
        values=[
            tf.expand_dims(bc_index, -1),
            tf.expand_dims(coord_hM, -1),
            tf.expand_dims(coord_wm, -1)
        ],
        axis=-1
    )  # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wm)
    coord_hMwM = tf.concat(
        values=[
            tf.expand_dims(bc_index, -1),
            tf.expand_dims(coord_hM, -1),
            tf.expand_dims(coord_wM, -1)
        ],
        axis=-1
    )  # [batch*i_c, i_h, i_w, f_h, f_w, 3] (batch*i_c, coord_hM, coord_wM)

    var_hmwm = tf.gather_nd(x_r, coord_hmwm)  # [batch*ic, i_h, i_w, f_h, f_w]
    var_hmwM = tf.gather_nd(x_r, coord_hmwM)  # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwm = tf.gather_nd(x_r, coord_hMwm)  # [batch*ic, i_h, i_w, f_h, f_w]
    var_hMwM = tf.gather_nd(x_r, coord_hMwM)  # [batch*ic, i_h, i_w, f_h, f_w]

    coord_hm = tf.cast(coord_hm, tf.float32)
    coord_hM = tf.cast(coord_hM, tf.float32)
    coord_wm = tf.cast(coord_wm, tf.float32)
    coord_wM = tf.cast(coord_wM, tf.float32)

    x_ip = var_hmwm*(coord_hM-coord_h)*(coord_wM-coord_w) + \
           var_hmwM*(coord_hM-coord_h)*(1-coord_wM+coord_w) + \
           var_hMwm*(1-coord_hM+coord_h)*(coord_wM-coord_w) + \
            var_hMwM*(1-coord_hM+coord_h)*(1-coord_wM+coord_w) # [batch*ic, ih, i_w, f_h, f_w]
    x_ip = tf.transpose(
        tf.reshape(x_ip, [i_c, batch, i_h, i_w, f_h, f_w]),
        [1, 2, 4, 3, 5, 0])  # [batch, i_h, f_h, i_w, f_w, i_c]
    x_ip = tf.reshape(x_ip, [batch, i_h * f_h, i_w * f_w,
                             i_c])  # [batch, i_h*f_h, i_w*f_w, i_c]
    with tf.variable_scope(scope or "deform_conv"):
        deform_conv = conv2d(
            x_ip,
            filter_shape,
            strides=[1, f_h, f_w, 1],
            activation=activation,
            scope="deform_conv")
    return deform_conv


"""
------------------------------------------------------------------
------------------------------------------------------------------
This is a tensorflow implementation of deformable convolution in
the paper Deformable Convolutional Networks.


by DouDou

2018-01-11
------------------------------------------------------------------
------------------------------------------------------------------

"""


class DeformConv2D(object):
    """

    Definition of DeformConv2D class

    """

    def __init__(self, x_, ks, co, name, groups, trainable):
        """

        Initialization.


        Params:
        --- x_: Input of the deformable convolutional layer, a 4-D
                Tensor with shape [bsize, height, width, channel].
        --- ks: Value of the kernel size.
        --- co: Output channels (Amount of kernels).
        --- name: Name of the deformable convolution layer.
        --- groups: The amount of groups.
        --- trainable: Whether the weights are trainable or not.

        """

        self.x = x_

        self.ks = ks

        self.co = co

        # Number of kernel elements
        self.N = ks**2

        self.name = name

        self.groups = groups

        self.trainable = trainable

    def conv(self, x_, co, mode, relu=True, groups=1, stride=1):
        """

        Definition of the regular 2D convolutional layer.


        Params:
        --- x_: Input of the convolutional layer, a 4-D Tensor with
                shape [bsize, height, width, channel].
        --- co: Output channels (Amount of kernels).
        --- mode: Purpose of convolution, "feature" or "offset".
        --- relu: Whether to apply the relu non-linearity or not.
        --- groups: The amount of groups.
        --- stride: Value of stride when doing convolution.
        Return:
        --- layer_output: Output of the convolutional layer.

        """

        # Ensure the mode is valid
        assert mode in ["feature", "offset"]

        with tf.name_scope(self.name + "_" + mode):

            # Get the kernel size
            ks = self.ks

            # Get the input channel
            ci = int(x_.get_shape()[-1]) // groups

            # Create the weights and biases
            if mode == "offset":

                #with tf.variable_scope(self.name + "_offset"):
                with tf.variable_scope(self.name):

                    # In offset mode, the weights are zero initialized
                    weights = tf.get_variable(
                        name="offset_weights",
                        shape=[ks, ks, ci, co],
                        trainable=self.trainable,
                        initializer=tf.zeros_initializer)

                    # Create the biases
                    biases = tf.get_variable(
                        name="offset_biases",
                        shape=[co],
                        trainable=self.trainable,
                        initializer=tf.zeros_initializer)

            else:

                with tf.variable_scope(self.name):

                    weights = tf.get_variable(
                        name="weights",
                        shape=[ks, ks, ci, co],
                        trainable=self.trainable)

                    # Create the biases
                    biases = tf.get_variable(
                        name="biases",
                        shape=[co],
                        trainable=self.trainable,
                        initializer=tf.zeros_initializer)

            # Define function for convolution calculation
            def conv2d(i_, w_):

                return tf.nn.conv2d(
                    i_, w_, [1, stride, stride, 1], padding="SAME")

            # If we don't need to divide this convolutional layer
            if groups == 1:

                layer_output = conv2d(x_, weights)

            # If we need to divide this convolutional layer
            else:

                # Split the input and weights
                group_inputs = tf.split(x_, groups, 3, name="split_input")

                group_weights = tf.split(
                    weights, groups, 3, name="split_weight")

                group_outputs = [
                    conv2d(i, w) for i, w in zip(group_inputs, group_weights)
                ]

                # Concatenate the groups
                layer_output = tf.concat(group_outputs, 3)

            # Add the biases
            layer_output = tf.nn.bias_add(layer_output, biases)

            if relu:

                # Nonlinear process
                layer_output = tf.nn.leaky_relu(layer_output)

            return layer_output

    def infer(self):
        """

        Function for deformable convolution.


        Return:
        --- layer_output: Output of the deformable convolutional layer.

        """

        with tf.name_scope(self.name):

            # Get the layer input
            x = self.x[:, :, :, :]

            # Get the kernel size.
            ks = self.ks

            # Get the number of kernel elements.
            # N = ks * ks, flatten
            N = self.N

            # Get the shape of the layer input.
            bsize, h, w, c = x.get_shape().as_list()

            # Get the offset, with shape [bsize, h, w, 2N].
            # [bsize, h, w, 2N] x and y offset in input location
            offset = self.conv(x, 2 * N, "offset", relu=False)

            # Get the data type of offset
            dtype = offset.dtype

            # pn ([1, 1, 1, 2N]) contains the locations in the kernel.
            # flatten kernel location
            pn = self.get_pn(dtype)

            # p0 ([1, h, w, 2N]) contains the location of each pixel on
            # the output feature map.
            p0 = self.get_p0([bsize, h, w, c], dtype)

            # p ([bsize, h, w, 2N]) contains the sample locations on the
            # input feature map of each pixel on the output feature map.
            p = p0 + pn + offset

            # Reshape p to [bsize, h, w, 2N, 1, 1].
            p = tf.reshape(p, [bsize, h, w, 2 * N, 1, 1])

            # q ([h, w, 2]) contains the location of each pixel on the
            # output feature map.
            q = self.get_q([bsize, h, w, c], dtype)

            # Get the bilinear interpolation kernel G ([bsize, h, w, N, h, w])
            # q[:, :, 0] => [h, w]
            # p[:, :, :, :N, :, :] - q[:, :, 0] broadcasting => [bsize, h, w, :N, h, w]
            gx = tf.maximum(1 - tf.abs(p[:, :, :, :N, :, :] - q[:, :, 0]), 0)

            gy = tf.maximum(1 - tf.abs(p[:, :, :, N:, :, :] - q[:, :, 1]), 0)

            G = gx * gy

            # Reshape G to [bsize, h*w*N, h*w]
            G = tf.reshape(G, [bsize, h * w * N, h * w])

            # Reshape x to [bsize, h*w, c]
            x = tf.reshape(x, [bsize, h * w, c])

            # Get x_offset
            # sigma(q) G * x(q)
            x_offset = tf.reshape(tf.matmul(G, x), [bsize, h, w, N, c])

            # Reshape x_offset to [bsize, h*kernel_size, w*kernel_size, c]
            x_offset = self.reshape_x_offset(x_offset, ks)

            # Get the output of the deformable convolutional layer
            layer_output = self.conv(
                x_offset,
                self.co,
                "feature",
                groups=self.groups,
                stride=ks,
                relu=False)

            return layer_output

    def get_pn(self, dtype):
        """

        Function to get pn.


        Params:
        --- dtype: Data type of pn.
        Return:
        --- pn: A 4-D Tensor with shape [1, 1, 1, 2N], which contains
                the locations in the kernel.

        """

        # Get the kernel size
        ks = self.ks

        pn_x, pn_y = np.meshgrid(
            range(-(ks - 1) // 2, (ks - 1) // 2 + 1),
            range(-(ks - 1) // 2, (ks - 1) // 2 + 1),
            indexing="ij")

        # The shape of pn is [2N,], order [x1, x2, ..., y1, y2, ...]
        pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))

        # Reshape pn to [1, 1, 1, 2N]
        pn = np.reshape(pn, [1, 1, 1, 2 * self.N])

        # Convert pn to TF Tensor
        pn = tf.constant(pn, dtype)

        return pn

    def get_p0(self, x_size, dtype):
        """

        Function to get p0.


        Params:
        --- x_size: Size of the input feature map.
        --- dtype: Data type of p0.
        Return:
        --- p0: A 4-D Tensor with shape [1, h, w, 2N], which contains
                the locations of each pixel on the output feature map.

        """

        # Get the shape of input feature map.
        bsize, h, w, c = x_size

        p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")

        p0_x = p0_x.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)

        p0_y = p0_y.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)

        p0 = np.concatenate((p0_x, p0_y), axis=3)

        # Convert p0 to TF Tensor
        p0 = tf.constant(p0, dtype)

        return p0

    def get_q(self, x_size, dtype):
        """

        Function to get q.


        Params:
        --- x_size: Size of the input feature map.
        --- dtype: Data type of q.
        Return:
        --- q: A 3-D Tensor with shape [h, w, 2], which contains the
               locations of each pixel on the output feature map.

        """

        # Get the shape of input feature map.
        bsize, h, w, c = x_size

        q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")

        q_x = q_x.flatten().reshape(h, w, 1)

        q_y = q_y.flatten().reshape(h, w, 1)

        q = np.concatenate((q_x, q_y), axis=2)

        # Convert q to TF Tensor
        q = tf.constant(q, dtype)

        return q

    @staticmethod
    def reshape_x_offset(x_offset, ks):
        """
        Function to reshape x_offset.


        Params:
        --- x_offset: A 5-D Tensor with shape [bsize, h, w, N, c].
        --- ks: The value of kernel size.
        Return:
        --- x_offset: A 4-D Tensor with shape [bsize, h*ks, w*ks, c].

        """

        # Get the shape of x_offset.
        bsize, h, w, N, c = x_offset.get_shape().as_list()

        # Get the new_shape
        new_shape = [bsize, h, w * ks, c]

        x_offset = [
            tf.reshape(x_offset[:, :, :, s:s + ks, :], new_shape)
            for s in range(0, N, ks)
        ]

        x_offset = tf.concat(x_offset, axis=2)

        # Reshape to final shape [bsize, h*ks, w*ks, c]
        x_offset = tf.reshape(x_offset, [bsize, h * ks, w * ks, c])

        return x_offset
