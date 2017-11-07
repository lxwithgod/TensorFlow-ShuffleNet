# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

__CONF = None


def parse_config(path):
    class Flag:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    global __CONF
    if not os.path.isfile(path):
        raise FileNotFoundError("Can not open file {}".format(path))

    with open(path) as f:
        __CONF = Flag(**yaml.load(f))


def get_config():
    global __CONF
    assert __CONF is not None, "Run parse_config first"
    return __CONF
