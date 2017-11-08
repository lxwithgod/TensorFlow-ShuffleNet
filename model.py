# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def get_model(image, classes, base_ch=144, groups=1, training=True):
    def shuffle_bottleneck(net, output, stride, group=1, scope="Unit"):
        assert 0 == output % group, "Output channels must be a multiple of groups"
        num_channels_in_group = output // group

        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': training}):
                if 1 != stride:
                    net_skip = slim.conv2d(net, output, [3, 3], 2, scope='3x3AVGPool')
                else:
                    net_skip = net
                net = slim.conv2d(net, output, [1, 1], activation_fn=tf.nn.relu, scope="1x1ConvIn")

                with tf.variable_scope("ChannelShuffle"):
                    net = tf.split(net, output, axis=3, name="split")
                    chs = []
                    for i in range(group):
                        for j in range(num_channels_in_group):
                            chs.append(tf.squeeze(net[i + j * group], name="squeeze"))
                    net = tf.stack(chs, axis=3, name="stack")

                with tf.variable_scope("3x3DWConv"):
                    depthwise_filter = tf.get_variable("depth_conv_w",
                                                       [3, 3, output, 1],
                                                       initializer=tf.truncated_normal_initializer())
                    net = tf.nn.depthwise_conv2d(net, depthwise_filter, [1, stride, stride, 1], 'SAME', name="DWConv")

                net = slim.conv2d(net, output, [1, 1], scope="1x1ConvOut")

                net = net + net_skip
        return net

    def shuffle_stage(net, output, repeat, group, scope="Stage"):
        with tf.variable_scope(scope):
            net = shuffle_bottleneck(net, output, 2, 1, scope='Unit{}'.format(0))
            for i in range(repeat):
                net = shuffle_bottleneck(net, output, 1, group, scope='Unit{}'.format(i + 1))
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
