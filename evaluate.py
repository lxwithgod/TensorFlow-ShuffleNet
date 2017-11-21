# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from dataset import ImageTXTDataset
from config import get_config, parse_config
from model import get_model


def main(_):
    conf = get_config()
    slim.get_or_create_global_step()

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
        tmp_cls = sorted([f for f in os.listdir(conf.data_train_dir)
                          if os.path.isdir(os.path.join(conf.data_train_dir, f))])
        class_map = {name: ith for ith, name in enumerate(tmp_cls)}
        dataset = ImageTXTDataset(os.path.join(conf.data_test_dir, 'images'),
                                  os.path.join(conf.data_test_dir, 'val_annotations.txt'),
                                  class_map=class_map,
                                  batch_size=1,
                                  pre_process_fn=pre_process_fn,
                                  one_hot=False,
                                  shuffle=conf.eval_random)
        image, label = dataset.get_data()
        label = tf.cast(label, tf.int64)

    with tf.variable_scope("ShuffleNet"):
        predictions = get_model(image,
                                classes=conf.classes,
                                shuffle=conf.shuffle,
                                base_ch=conf.base_ch,
                                groups=conf.groups)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'top_1_accuracy':
            slim.metrics.streaming_mean(tf.nn.in_top_k(predictions, label, 1)),
        'top_{}_accuracy'.format(conf.show_top_k):
            slim.metrics.streaming_mean(tf.nn.in_top_k(predictions, label, conf.show_top_k)),
    })

    summary_ops = []
    for metric_name, metric_value in names_to_values.items():
        op = tf.summary.scalar(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    if conf.watch:
        slim.evaluation.evaluation_loop(
            '',
            log_dir,
            log_dir,
            num_evals=conf.eval_batch,
            eval_op=list(names_to_updates.values()),
            summary_op=tf.summary.merge(summary_ops),
            eval_interval_secs=conf.eval_interval)
    else:
        assert conf.ckpt is not None, "You should specify the '.ckpt' file"
        slim.evaluation.evaluate_once(
            '',
            conf.ckpt,
            log_dir,
            num_evals=conf.eval_batch,
            eval_op=list(names_to_updates.values()),
            summary_op=tf.summary.merge(summary_ops))


if '__main__' == __name__:
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', default='conf/demo.yml', help='Path to the config file')
    parser.add_argument('--watch', '-w', action='store_true', help='Run-once (default), or watch')
    parser.add_argument('--ckpt', '-f', help='Checkpoint file for run-once mode')
    args = parser.parse_args()
    parse_config(args.conf, watch=args.watch, ckpt=args.ckpt)

    tf.app.run()
