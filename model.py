# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_model(image, classes, base_ch=144, groups=1, training=True):
    def shuffle_bottleneck(net, output, stride, group=1, scope="Unit"):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': training}):
                if 1 != stride:
                    net_skip = slim.conv2d(net, output, [3, 3], 2)
                else:
                    net_skip = net

                net = slim.conv2d(net, output, [1, 1], activation_fn=tf.nn.relu)

                depthwise_filter = tf.get_variable("depth_conv_w",
                                                   [3, 3, output, 1],
                                                   initializer=tf.truncated_normal_initializer())
                net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")

                net = slim.conv2d(net, output, [1, 1])

                net = net + net_skip
        return net

    def shuffle_stage(net, output, repeat, group, scope="Stage"):
        with tf.variable_scope(scope):
            with tf.variable_scope('Unit{}'.format(0)):
                net = shuffle_bottleneck(net, output, 2, 1)
            for i in range(repeat):
                with tf.variable_scope('Unit{}'.format(i + 1)):
                    net = shuffle_bottleneck(net, output, 1, group)
        return net

    with slim.arg_scope([slim.conv2d]):
        with tf.variable_scope('Stage1'):
            net = slim.conv2d(image, 24, [3, 3], 2)
            net = slim.max_pool2d(net, [3, 3], 2)

        net = shuffle_stage(net, base_ch, 3, groups, 'Stage2')
        net = shuffle_stage(net, base_ch * 2, 7, groups, 'Stage3')
        net = shuffle_stage(net, base_ch * 4, 3, groups, 'Stage4')

        with tf.variable_scope('Stage5'):
            net = tf.reduce_mean(net, [1, 2])
            net = slim.fully_connected(net, classes)

        return net
