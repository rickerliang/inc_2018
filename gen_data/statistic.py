#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from dicttoxml import dicttoxml

SUPERCATEGORY = 'supercategory'
KINGDOM = 'kingdom'
PHYLUM = 'phylum'
CLASS = 'class'
ORDER = 'order'
FAMILY = 'family'
GENUS = 'genus'
ID = 'id'

CATEGORY_COUNT = 8142
TRAIN_IMAGE_COUNT = 437513
VAL_IMAGE_COUNT = 24426

def parse(json_file):
    with open(json_file, 'r') as f:
        json_str = f.read()
        json_object = json.loads(json_str)
        return json_object

def add_image_count_by_category_name(name, tree):
    tree[name]['image_count'] += 1
    return tree[name]['sub_cate']


def add_image_count(c, categories_tree):

    kingdom_name = c[KINGDOM]
    phylum_name = c[PHYLUM]
    class_name = c[CLASS]
    order_name = c[ORDER]
    family_name = c[FAMILY]
    genus_name = c[GENUS]
    id_name = str(c[ID])

    tree = add_image_count_by_category_name(kingdom_name, categories_tree)

    tree = add_image_count_by_category_name(phylum_name, tree)

    tree = add_image_count_by_category_name(class_name, tree)

    tree = add_image_count_by_category_name(order_name, tree)

    tree = add_image_count_by_category_name(family_name, tree)

    tree = add_image_count_by_category_name(genus_name, tree)

    tree[id_name]['image_count'] += 1

def build_annotation(json_object, categories_tree):
    for a in json_object['annotations']:
        category_id = a['category_id']
        c = json_object['categories'][category_id]

        add_image_count(c, categories_tree)

    return

def build_categories_tree(json_file):

    categories_tree = dict()

    json_object = parse(json_file)
    for c in json_object['categories']:
        kingdom_name = c[KINGDOM]
        phylum_name = c[PHYLUM]
        class_name = c[CLASS]
        order_name = c[ORDER]
        family_name = c[FAMILY]
        genus_name = c[GENUS]
        id_name = str(c[ID])

        phylum_tree = fill_tree(categories_tree, kingdom_name)

        class_tree = fill_tree(phylum_tree, phylum_name)

        order_tree = fill_tree(class_tree, class_name)

        family_tree = fill_tree(order_tree, order_name)

        genus_tree = fill_tree(family_tree, family_name)

        id_tree = fill_tree(genus_tree, genus_name)

        if not (id_name in id_tree):
            id_tree[id_name] = {'cate_count': 0, 'image_count': 0}
        else:
            print('*O*')

            id_tree[id_name]['cate_count'] += 1

    build_annotation(json_object, categories_tree)
    return categories_tree


def fill_tree(categories_tree, name):
    if not (name in categories_tree):
        # subcategories, subcategory count, image count
        categories_tree[name] = {'sub_cate': dict(), 'cate_count': 0, 'image_count': 0}
    category = categories_tree[name]
    leaf_tree = category['sub_cate']
    category['cate_count'] += 1
    return leaf_tree


if __name__ == "__main__":
    train_categories_tree = build_categories_tree('raw_data/train2018.json')
    val_categories_tree = build_categories_tree('raw_data/val2018.json')

    train_image_count = 0
    train_categories = 0
    for kingdom, leaf_tree in train_categories_tree.items():
        train_categories += leaf_tree['cate_count']
        train_image_count += leaf_tree['image_count']

    val_image_count = 0
    val_categories = 0
    for kingdom, leaf_tree in val_categories_tree.items():
        val_categories += leaf_tree['cate_count']
        val_image_count += leaf_tree['image_count']


    assert(train_categories == val_categories)
    assert(train_image_count == TRAIN_IMAGE_COUNT)
    assert(val_image_count == VAL_IMAGE_COUNT)

    xml = dicttoxml(train_categories_tree)
    with open('train_tree.xml', 'w') as f:
        f.write(xml)

    xml = dicttoxml(val_categories_tree)
    with open('val_tree.xml', 'w') as f:
        f.write(xml)
