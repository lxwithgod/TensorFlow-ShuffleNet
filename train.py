# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from dataset import ImageFolderDataset
from config import get_config, parse_config
from model import get_model


def main(_):
    conf = get_config()
    global_step = slim.get_or_create_global_step()

    log_dir = os.path.join(conf.log_dir, conf.name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir, 0o755)

    def pre_process_fn(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 1.2)
        image = tf.image.random_saturation(image, 0.3, 1)
        image = tf.image.random_hue(image, 0.25)
        image = tf.image.resize_images(image, [conf.image_height, conf.image_width])
        return image

    with tf.variable_scope("dataset"):
        dataset = ImageFolderDataset(conf.data_train_dir,
                                     batch_size=conf.batch_size,
                                     epoch=conf.epoch,
                                     pre_process_fn=pre_process_fn,
                                     n_threads=conf.read_thread,
                                     one_hot=True)
        image, label = dataset.get_data()

        tf.summary.image('input_image', image, max_outputs=4)
        class_label = tf.argmax(label, axis=1)

    with tf.variable_scope("ShuffleNet"):
        predictions = get_model(image,
                                classes=conf.classes,
                                shuffle=conf.shuffle,
                                base_ch=conf.base_ch,
                                groups=conf.groups)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=predictions)

    learning_rate = tf.train.exponential_decay(conf.learning_rate, global_step, 1000, 0.95)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=conf.momentum)

    train_op = tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer)

    with tf.variable_scope("Summary"):
        accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions, class_label, 1))
        tf.summary.scalar('accuracy', accuracy[1])
        accuracy_top_k = tf.metrics.mean(tf.nn.in_top_k(predictions, class_label, conf.show_top_k))
        tf.summary.scalar('accuracy_top_{}'.format(conf.show_top_k), accuracy_top_k[1])

        for l in tf.trainable_variables():
            tf.summary.histogram(l.name.replace(':', "_"), l)

    slim.learning.train(
        train_op,
        log_dir,
        log_every_n_steps=10,
        save_summaries_secs=10,
        save_interval_secs=conf.save_interval)


if '__main__' == __name__:
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', default='conf/demo.yml', help='Path to the config file')
    args = parser.parse_args()
    parse_config(args.conf)

    tf.app.run()
