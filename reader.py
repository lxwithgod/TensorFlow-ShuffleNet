# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
from os import listdir
from os.path import isfile, join, isdir


class Dataset():
    def __init__(self,
                 batch_size=4,
                 epoch=1,
                 pre_process_fn=None,
                 shuffle=True,
                 one_hot=False,
                 n_cpus=4):
        self.batch_size = batch_size
        self.epoch = epoch
        self.pre_process_fn = pre_process_fn
        self.shuffle = shuffle
        self.one_hot = one_hot
        self.n_cpus = n_cpus

        self._dataset = None
        self.n_classes = 0
        self.len = 0
        self.classes = None
        self.prepare()

    def prepare(self):
        pass

    def get_data(self):
        assert self._dataset is not None, "Dataset not prepared"
        iterator = self._dataset.make_one_shot_iterator()
        return iterator.get_next()

    def _shuffle_data_and_label(self, data, label):
        data_and_label = [(f, l) for f, l in zip(data, label)]
        random.shuffle(data_and_label)
        data = [d[0] for d in data_and_label]
        label = [l[1] for l in data_and_label]
        return data, label

    def _parse_function(self, filename, clazz):
        image_string = tf.read_file(filename)
        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.pre_process_fn is not None:
            image = self.pre_process_fn(image)
        if self.one_hot:
            clazz = tf.one_hot(clazz, self.n_classes)
        return image, clazz


class ImageFolderDataset(Dataset):
    def __init__(self, root, **args):
        self.root = root
        super(ImageFolderDataset, self).__init__(**args)

    def prepare(self):
        self.classes = sorted([f for f in listdir(self.root) if isdir(join(self.root, f))])
        self.n_classes = len(self.classes)
        filenames = []
        clazz = []
        for ith, fo in enumerate(self.classes):
            images = [join(self.root, fo, 'images', f)
                      for f in listdir(join(self.root, fo, 'images'))
                      if isfile(join(self.root, fo, 'images', f))]
            if not self.shuffle:
                images = sorted(images)
            filenames.extend(images)
            clazz.extend([ith] * len(images))

        self.len = len(filenames)

        if self.shuffle:
            self._shuffle_data_and_label(filenames, clazz)

        self._dataset = tf.data.Dataset.from_tensor_slices((filenames, clazz))
        self._dataset = self._dataset.map(self._parse_function, num_parallel_calls=self.n_cpus)
        self._dataset = self._dataset.batch(self.batch_size)
        self._dataset = self._dataset.repeat(self.epoch)
        self._dataset = self._dataset.prefetch(self.batch_size * self.n_cpus)

        if self.shuffle:
            self._dataset = self._dataset.shuffle(self.batch_size * self.n_cpus)
