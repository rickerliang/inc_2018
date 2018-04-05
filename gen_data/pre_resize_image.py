#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from PIL import Image

from gen_data.data_provider import *


def resize(dataset_file):
    size = (512, 512)
    filenames, _, _ = read_labeled_image_list(dataset_file, 0)
    for i, filename in enumerate(filenames):
        im = Image.open(filename)
        im = im.resize(size, Image.ANTIALIAS)
        im.save(filename)
        sys.stdout.write("\r{0}".format(i))


if __name__ == "__main__":
    resize("_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_val__7_dataset.txt")