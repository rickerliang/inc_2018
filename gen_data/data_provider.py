#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import math
import os

import tensorflow as tf

"""
1.build category directory
2.use gen_dataset_file(train_file_dir, depth) generate train/val file
3.use get_tf_dataset(train/val_file, balance_count, parallel_call) build tensorflow input pipeline
"""

def read_labeled_image_list(dataset_text_file, balance_count):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(dataset_text_file, 'r')

    dict = {}

    for line in f:
        filename, label = line.rstrip().split('\t')
        label = int(label)

        if label in dict:
            dict[label].append(filename)
        else:
            dict[label] = [filename]

    for key in dict.keys():
        while len(dict[key]) < balance_count:
            dict[key] += dict[key]

    filenames = []
    labels = []
    for key in dict.keys():
        filenames += dict[key]
        key = [key] * len(dict[key])
        labels += key

    return filenames, labels


def get_tf_dataset(dataset_text_file, balance_count=500, parallel_calls=20, resize=512, crop_size=384):
    def aug_1(image):
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        return image

    def aug_2(image):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        return image

    def aug_3(image):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        return image

    def aug_4(image):
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        return image

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [resize, resize])

        image_flipped = tf.image.random_flip_left_right(image_resized)

        angle = tf.reshape(tf.random_uniform([1], -math.pi/12, math.pi/12, tf.float32), [])
        image_rotated = tf.contrib.image.rotate(image_flipped, angle, interpolation='BILINEAR')


        image_cropped = tf.random_crop(image_rotated, [crop_size, crop_size, 3])

        p1 = partial(aug_1, image_cropped)
        p2 = partial(aug_2, image_cropped)
        p3 = partial(aug_3, image_cropped)
        p4 = partial(aug_4, image_cropped)

        k = tf.reshape(tf.random_uniform([1], 0, 4, tf.int32), [])
        image = tf.case([(tf.equal(k, 0), p1),
                         (tf.equal(k, 1), p2),
                         (tf.equal(k, 2), p3),
                         (tf.equal(k, 3), p4)],
                        default=p1,
                        exclusive=True)

        return image, label

    filenames, labels = read_labeled_image_list(dataset_text_file, balance_count)
    filenames = tf.constant(filenames, name='filename_list')
    labels = tf.constant(labels, name='label_list')

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function, num_parallel_calls=parallel_calls)
    #dataset = dataset.cache()
    dataset = dataset.prefetch(10000)

    return dataset


def collect_all_files(folder, label):
    """
    collect file in 'folder' (also collect in subfolder of 'folder')
    :param folder:
    :return:
    """
    ret = []
    for root, subfolders, files in os.walk(folder):
        for file in files:
            ret.append((os.path.join(root, file), label))

    return ret


def list_subdir_only(dir):
    return [os.path.join(dir, name) for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def build_candidate_dir_list(root_dir, depth):
    result = [root_dir]
    for i in range(depth):
        subdir = []
        for s in result:
            subdir += list_subdir_only(s)

        result = subdir

    return result


def collect_file_path_and_label(root_dir, depth):
    """
    note, all leaf nodes must have the same depth
    :param root_dir:
    :param depth:
    :return: [(image_full_path, int_label), (image_full_path, int_label)],
            {int_label: (folder_path, count), int_label: (folder_path, count)}

    use parameter 'depth' to control category granularity

    here is the example:

    depth:
        0               1                   2                   3                   4
    directory structure:
    root_folder -- level_0_folder_0 -- level_1_folder_0 -- level_2_folder_0 -- level_3_folder_0 -- image_a
                |                   |                   |                   |                   |- image_b
                |                   |                   |                   |                   |- image_c
                |                   |                   |                   |
                |                   |                   |                   |- level_3_folder_1 -- image_d
                |                   |                   |                                       |- image_e
                |                   |                   |
                |                   |                   |- level_2_folder_1 -- level_3_folder_2 -- image_f
                |                   |                                       |- level_3_folder_3 -- image_g
                |                   |
                |                   |- level_1_folder_1 -- level_2_folder_2 -- level_3_folder_4 -- image_h
                |                                       |                   |                   |- image_i
                |                                       |                   |                   |- image_j
                |                                       |                   |
                |                                       |                   |- level_3_folder_5 -- image_k
                |                                       |                                       |- image_l
                |                                       |
                |                                       |- level_2_folder_3 -- level_3_folder_6 -- image_m
                |                                                           |- level_3_folder_7 -- image_n
                |
                |- level_0_folder_1 -- level_1_folder_2 -- level_2_folder_4 -- level_3_folder_8 -- image_o
                                    |                   |                   |                   |- image_p
                                    |                   |                   |                   |- image_q
                                    |                   |                   |                             
                                    |                   |                   |- level_3_folder_9 -- image_r
                                    |                   |                                       |- image_s
                                    |                   |                                                 
                                    |                   |- level_2_folder_5 -- level_3_folder_10 -- image_t
                                    |                                       |- level_3_folder_11 -- image_u
                                    |                                                                     
                                    |- level_1_folder_3 -- level_2_folder_6 -- level_3_folder_12 -- image_v
                                                        |                   |                    |- image_w
                                                        |                   |                    |- image_x
                                                        |                   |                             
                                                        |                   |- level_3_folder_13 -- image_y
                                                        |                                        |- image_z
                                                        |                                                 
                                                        |- level_2_folder_7 -- level_3_folder_14 -- image_α
                                                                            |- level_3_folder_15 -- image_β

    case 1: root_dir = root_folder
            depth = 0
            category = 1
            categories[0].images = [image_a : image_β]

    case 2: root_dir = root_folder
            depth = 3
            category = 8
            categories[2].images = [image_h : image_l]

    case 3: root_dir = level_1_folder_2
            depth = 2
            category = 4
            categories[2].image = [image_m]

    """

    ret_list = []
    ret_dict = {}

    candidate_dir_list = build_candidate_dir_list(root_dir, depth)
    for i, dir in enumerate(candidate_dir_list):
        file_list = collect_all_files(dir, i)
        ret_list += file_list
        ret_dict[i] = (dir, len(file_list))


    return ret_list, ret_dict


def gen_dataset_file(raw_data_root_dir, depth):
    file_label_list, label_dir_dict = collect_file_path_and_label(raw_data_root_dir, depth)
    raw_data_root_dir = raw_data_root_dir.replace('/', '_')

    with open('{0}_dataset_file_{1}.txt'.format(raw_data_root_dir, depth), 'w') as f:
        for file, label in file_label_list:
            f.write('{0}\t{1}\n'.format(file, label))

    total_file = 0
    with open('{0}_dict_file_{1}.txt'.format(raw_data_root_dir, depth), 'w') as f:
        for key, item in label_dir_dict.iteritems():
            f.write('{0}\t{1}\t{2}\n'.format(key, item[0], item[1]))
            total_file += item[1]

    return total_file


