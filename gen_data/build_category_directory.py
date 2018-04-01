#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import errno
import sys

from gen_data.statistic import *

# working dir is iNaturalist_2018_Competition/

SOURCE = '../iNaturalist_image/'
DEST_TRAIN = '../iNaturalist_image/train/'
DEST_VAL = '../iNaturalist_image/val/'
TRAIN_JSON_PATH = 'raw_data/train2018.json'
VAL_JSON_PATH = 'raw_data/val2018.json'


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def build_category_directory(dest_root_path, source_json_path):
    if not os.path.exists(dest_root_path):
        os.makedirs(dest_root_path)
    json_object = parse(source_json_path)
    annotations = json_object['annotations']
    categories = json_object['categories']

    categories_string_dict = {}
    for category in categories:
        kingdom_name = category[KINGDOM]
        phylum_name = category[PHYLUM]
        class_name = category[CLASS]
        order_name = category[ORDER]
        family_name = category[FAMILY]
        genus_name = category[GENUS]
        id_name = category[ID]

        categories_string_dict['{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(kingdom_name,
                                                        phylum_name,
                                                        class_name,
                                                        order_name,
                                                        family_name,
                                                        genus_name,
                                                        id_name)] = 0

    dict_for_check = {}

    for i, image_desc in enumerate(json_object['images']):
        image_file_name = image_desc['file_name']
        image_id = image_desc['id']

        category_id = annotations[i]['category_id']
        assert(image_id == annotations[i]['image_id'])
        c = categories[category_id]

        super_category_name = c[SUPERCATEGORY]

        kingdom_name = c[KINGDOM]
        phylum_name = c[PHYLUM]
        class_name = c[CLASS]
        order_name = c[ORDER]
        family_name = c[FAMILY]
        genus_name = c[GENUS]
        id_name = c[ID]

        category_key = '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(kingdom_name,
                                         phylum_name,
                                         class_name,
                                         order_name,
                                         family_name,
                                         genus_name, id_name)
        dict_for_check[category_key] = 0
        categories_string_dict.pop(category_key, None)

        image_file_name = os.path.abspath(os.path.join(SOURCE, image_file_name))

        dest_path = os.path.join(dest_root_path, kingdom_name,
                                 phylum_name, class_name, order_name,
                                 family_name, genus_name, str(id_name))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        dest_file_name = os.path.abspath(os.path.join(dest_path, os.path.basename(image_file_name)))

        symlink_force(image_file_name, dest_file_name)
        if i % 50000 == 0:
            sys.stdout.write('*')

    assert(len(categories_string_dict) == 0)

    return len(dict_for_check)


if __name__ == "__main__":
    assert(build_category_directory(DEST_TRAIN, TRAIN_JSON_PATH) == CATEGORY_COUNT)
    assert(build_category_directory(DEST_VAL, VAL_JSON_PATH) == CATEGORY_COUNT)
