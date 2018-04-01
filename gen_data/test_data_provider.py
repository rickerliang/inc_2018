#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf

from gen_data.data_provider import *

class DataProviderTestCase(unittest.TestCase):
    # def test_collect_all_files(self):
    #     ret = collect_all_files('unittest_data/level_0_folder_0/level_1_folder_0', 0)
    #     self.assertEqual(len(ret), 7)
    #
    #     ret = collect_all_files('unittest_data/level_0_folder_1', 1)
    #     self.assertEqual(len(ret), 14)
    #
    # def test_build_candidate_dir_list(self):
    #     ret = build_candidate_dir_list('unittest_data', 2)
    #     self.assertEqual(len(ret), 4)
    #
    #     ret = build_candidate_dir_list('unittest_data', 3)
    #     self.assertEqual(len(ret), 8)
    #
    # def test_collect_file_path_and_label(self):
    #     file_list, label_dict = collect_file_path_and_label('unittest_data', 2)
    #     self.assertEqual(len(label_dict), 4)

    def test_read_labeled_image_list(self):
        dataset = get_tf_dataset(
            "_home_lyk_machine_learning_Supervised_Learning_iNaturalist_image_val__dataset_file_7.txt", 20)

        dataset.repeat()
        #dataset.shard(2, 0)
        dataset.shuffle(10000)

        batched_dataset = dataset.batch(4)
        iterator = batched_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        with tf.Session(config=session_config) as sess:
            images, labels = sess.run(next_element)

        self.assertEqual(images.shape[0], 4)
        self.assertEqual(labels.shape[0], 4)

if __name__ == '__main__':
    unittest.main()
