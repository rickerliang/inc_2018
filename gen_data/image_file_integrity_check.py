#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from PIL import Image

from gen_data.data_provider import *


def verify(filename_list):
    file_count = 0
    for filename in filename_list:
        im = Image.open(filename)
        file_count += 1
        sys.stdout.write("\r{0}".format(file_count))
        try:
            im.verify()
        except Exception as e:
            print(filename, str(e))

if __name__ == "__main__":
    train_filenames, _, _ = read_labeled_image_list("_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_train__dataset_file_7.txt", 0)
    verify(train_filenames)
    val_filenames, _, _ = read_labeled_image_list("_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_val__dataset_file_7.txt", 0)
    verify(val_filenames)
