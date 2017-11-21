# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import xmltodict
from os import listdir
from os.path import isfile, join, isdir, splitext


class Dataset:
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
        if self.class_map is not None:
            self.n_classes = len(self.class_map.keys())
        else:
            self.n_classes = 0
        self.len = 0

        # Should do
        self._prepare()
        self._apply_settings()

    def get_data(self):
        assert self._dataset is not None, "Dataset not prepared"
        iterator = self._dataset.make_one_shot_iterator()
        return iterator.get_next()

    def _prepare(self):
        pass

    def _apply_settings(self):
        if 0 == self.n_classes:
            self.n_classes = len(self.class_map)
        self._dataset = self._dataset.map(self._parse_function, num_parallel_calls=self.n_threads)
        self._dataset = self._dataset.batch(self.batch_size)
        self._dataset = self._dataset.repeat(self.epoch)
        self._dataset = self._dataset.prefetch(self.batch_size * self.n_threads)

        if self.shuffle:
            self._dataset = self._dataset.shuffle(self.batch_size * self.n_threads)

    def _make_class_map(self, clazz):
        self.class_map = {name: ith for ith, name in enumerate(sorted(list(set(clazz))))}

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

    def _prepare(self):
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

        # Shuffle
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

    def _prepare(self):
        filenames = []
        clazz = []

        # Search all files
        for line in open(self.txt_file):
            line = line.split()
            filenames.append(join(self.root, line[0].strip()))
            clazz.append(line[1].strip())

        # Make classmap
        if self.class_map is None:
            self._make_class_map(clazz)

        # Convert class name to id
        clazz = [self.class_map[c] for c in clazz]

        # Shuffle
        if self.shuffle:
            filenames, clazz = self._shuffle_data_and_label(filenames, clazz)

        # Make dataset
        self.len = len(filenames)
        self._parse_function = self._parse_function_path
        self._dataset = tf.data.Dataset.from_tensor_slices((filenames, clazz))


class ImageNetClsDataset(Dataset):
    def __init__(self, root, split='train', n_classes=1000, **args):
        assert split in ['train', 'val', 'test'], "Unknown split"
        self.root = root
        self.split = split
        self.n_classes = n_classes

        super().__init__(**args)

    def _prepare(self):
        filenames = []
        clazz = []

        image_dir = join(self.root, 'Data', 'CLS-LOC', self.split)
        if 'train' == self.split:
            """
            Training set is a Image Folder Dataset
            """
            for folder in listdir(image_dir):
                images = [f for f in listdir(join(image_dir, folder))]
                images = [join(image_dir, folder, f) for f in images]
                images = [f for f in images if isfile(f)]

                if not self.shuffle:
                    images = sorted(images)

                filenames.extend(images)
                clazz.extend([folder] * len(images))

        elif 'val' == self.split:
            """
            Validation set 
            """
            annotation_dir = join(self.root, 'Annotations', 'CLS-LOC', self.split)
            for f in listdir(image_dir):
                fnm, fex = splitext(f)
                imf = join(image_dir, f)
                anf = join(annotation_dir, fnm + ".xml")
                with open(anf, 'r') as an:
                    anno = xmltodict.parse(an.read())
                    objs = anno['annotation']['object']
                    if not isinstance(objs, list):
                        objs = [objs]
                    cls = None
                    for o in objs:
                        if cls is None:
                            cls = o['name']
                        else:
                            assert cls == o['name'], 'One image contains conflicting labels'
                    filenames.append(imf)
                    clazz.append(cls)

        elif 'test' == self.split:
            """
            Test set do not have labels
            """
            images = listdir(image_dir)
            images = [join(image_dir, f) for f in images]
            images = [f for f in images if isfile(f)]
            clazz = [None] * len(images)
        else:
            raise ValueError('Unknown split')

        if self.split in ['train', 'val']:
            if self.class_map is None:
                self._make_class_map(clazz)
                if 0 != self.n_classes:
                    assert len(self.class_map.keys()) == self.n_classes, 'Some class have 0 samples'

            # Convert class name to id
            clazz = [self.class_map[c] for c in clazz]

        # Shuffle
        if self.shuffle:
            filenames, clazz = self._shuffle_data_and_label(filenames, clazz)

        # Make dataset
        self.len = len(filenames)
        self._parse_function = self._parse_function_path
        self._dataset = tf.data.Dataset.from_tensor_slices((filenames, clazz))
