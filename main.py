# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from reader import get_image_folder_dataset
from config import get_config, parse_config
from model import get_model


def main(_):
    conf = get_config()

    def pre_process_fn(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 1.2)
        image = tf.image.random_saturation(image, 0.3, 1)
        image = tf.image.random_hue(image, 0.25)
        image = tf.image.resize_images(image, [conf.image_height, conf.image_width])
        return image

    with tf.variable_scope("dataset"):
        dataset, classes = get_image_folder_dataset(conf.data_dir,
                                                    batch_size=conf.batch_size,
                                                    epoch=conf.epoch,
                                                    pre_process_fn=pre_process_fn,
                                                    one_hot=True)
        image, label = dataset.get_next()
        class_label = tf.argmax(label, axis=1)

    with tf.variable_scope("MyNet"):
        predictions = get_model(image,
                                classes=conf.classes,
                                shuffle=conf.shuffle,
                                base_ch=conf.base_ch,
                                groups=conf.groups)

    with tf.variable_scope("Summary"):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=predictions)
        tf.summary.scalar('loss', loss)
        accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions, class_label, 1))
        tf.summary.scalar('accuracy', accuracy[1])
        accuracy_top_k = tf.metrics.mean(tf.nn.in_top_k(predictions, class_label, conf.show_top_k))
        tf.summary.scalar('accuracy_top_{}'.format(conf.show_top_k), accuracy_top_k[1])

    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    train_op = slim.learning.create_train_op(loss, optimizer)

    slim.learning.train(
        train_op,
        conf.log_dir,
        log_every_n_steps=10,
        save_summaries_secs=10,
        save_interval_secs=600)


if '__main__' == __name__:
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', default='conf/demo.yml', help='Path to the config file')
    args = parser.parse_args()
    parse_config(args.conf)

    tf.app.run()
