# -*- coding: utf-8 -*-

import numpy as np
import h5dy
import json
import tensorflow as tf

[IMG_HEIGTH, IMG_WIDTH] = [224, 224]

input_json = './data/prepro.json'
image_root = './data'

batch_size = 16


image_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_feature_extract_model = tf.keras.Model(new_input, hidden_layer)

data = json.load(open(input_json, 'r'))


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_HEIGTH, IMG_WIDTH))
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img, image_path)


def extract_feat(img_list, name):
    dateLen = len(img_list)


extract_feat(data['unique_img_train'], 'images_train')
extract_feat(data['unique_img_test'], 'images_test')
