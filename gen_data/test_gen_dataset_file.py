#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from unittest import TestCase

from gen_data.data_provider import *
from gen_data.statistic import *


class TestGen_dataset_file(TestCase):
    def test_gen_dataset_file(self):
        depth = 7
        self.assertEqual(TRAIN_IMAGE_COUNT,
                         gen_dataset_file('/home/lyk/machine_learning/Supervised_Learning/iNaturalist_image/train/', depth))

        self.assertEqual(VAL_IMAGE_COUNT,
                         gen_dataset_file('/home/lyk/machine_learning/Supervised_Learning/iNaturalist_image/val/', depth))
