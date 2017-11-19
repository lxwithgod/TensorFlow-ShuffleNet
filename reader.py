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
                 class_map=None,
                 pre_process_fn=None,
                 shuffle=True,
                 one_hot=False,
                 n_threads=4):
        self.batch_size = batch_size
        self.epoch = epoch
        self.class_map = class_map
        self.pre_process_fn = pre_process_fn
        self.shuffle = shuffle
        self.one_hot = one_hot
        self.n_threads = n_threads

        # Need to be initialized
        self._parse_function = None
        self._dataset = None
        self.n_classes = 0
        self.len = 0

        # Should do
        self.prepare()
        self.apply_settings()

    def apply_settings(self):
        self.n_classes = len(self.class_map)
        self._dataset = self._dataset.map(self._parse_function, num_parallel_calls=self.n_threads)
        self._dataset = self._dataset.batch(self.batch_size)
        self._dataset = self._dataset.repeat(self.epoch)
        self._dataset = self._dataset.prefetch(self.batch_size * self.n_threads)

        if self.shuffle:
            self._dataset = self._dataset.shuffle(self.batch_size * self.n_threads)

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

    def _parse_function_path(self, filename, clazz):
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
        super().__init__(**args)

    def prepare(self):
        # Generate class id map
        if self.class_map is None:
            tmp_cls = sorted([f for f in listdir(self.root) if isdir(join(self.root, f))])
            self.class_map = {name: ith for ith, name in enumerate(tmp_cls)}

        # Search all files
        filenames = []
        clazz = []
        for folder, cls in self.class_map.items():
            images = [join(self.root, folder, 'images', f)
                      for f in listdir(join(self.root, folder, 'images'))
                      if isfile(join(self.root, folder, 'images', f))]
            if not self.shuffle:
                images = sorted(images)
            filenames.extend(images)
            clazz.extend([cls] * len(images))

        # Shuffile
        if self.shuffle:
            filenames, clazz = self._shuffle_data_and_label(filenames, clazz)

        # Make dataset
        self.len = len(filenames)
        self._parse_function = self._parse_function_path
        self._dataset = tf.data.Dataset.from_tensor_slices((filenames, clazz))


class ImageTXTDataset(Dataset):
    def __init__(self, root, txt_file, **args):
        self.root = root
        self.txt_file = txt_file
        super().__init__(**args)

    def prepare(self):
        filenames = []
        clazz = []

        # Search all files
        for line in open(self.txt_file):
            line = line.split()
            filenames.append(join(self.root, line[0].strip()))
            clazz.append(line[1].strip())

        # Make classmap
        if self.class_map is None:
            self.class_map = {name: ith for ith, name in enumerate(sorted(list(set(clazz))))}

        # Convert class name to id
        clazz = [self.class_map[c] for c in clazz]

        # Shuffle
        if self.shuffle:
            filenames, clazz = self._shuffle_data_and_label(filenames, clazz)

        # Make dataset
        self.len = len(filenames)
        self._parse_function = self._parse_function_path
        self._dataset = tf.data.Dataset.from_tensor_slices((filenames, clazz))
