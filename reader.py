# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
from os import listdir
from os.path import isfile, join, isdir


def get_image_folder_dataset(root, batch_size=4, epoch=1, pre_process_fn=None, is_training=True, one_hot=False):
    """ 直接读取整个文件夹作为数据集 """

    def _parse_function(filename, clazz):
        image_string = tf.read_file(filename)
        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if pre_process_fn is not None:
            image = pre_process_fn(image)
        if one_hot:
            clazz = tf.one_hot(clazz, n_classes)
        return image, clazz

    classes = sorted([f for f in listdir(root) if isdir(join(root, f))])
    n_classes = len(classes)
    filenames = []
    clazz = []
    for ith, fo in enumerate(classes):
        images = [join(root, fo, 'images', f)
                  for f in listdir(join(root, fo, 'images'))
                  if isfile(join(root, fo, 'images', f))]
        if not is_training:
            images = sorted(images)
        filenames.extend(images)
        clazz.extend([ith] * len(images))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, clazz))
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.prefetch(batch_size * 4)

    if is_training:
        dataset = dataset.shuffle(batch_size * 4)

    iterator = dataset.make_one_shot_iterator()

    return iterator, classes
