# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import json


###################
# global config  #
##################

print("Loading config")
# data input
train_input_json = "./data/train_data.json"
test_input_json = "./data/test_data.json"

# train params
lr = 0.0003
lr_decay_start = -1             # when begin to decay lr(-1 never)
batch_size = 128
buffer_size = 1000
input_embedding_size = 1024     # encoding size of each token in vocab
rnn_size = 256                  # node# each rnn layer
rnn_layer = 2
dim_image = 512
dim_hidden = 1024               # size of common embedding vector
dim_attention = 512             # size of attention embedding
num_output = 2000               # output answer number
img_norm = 1                    # whether norm image feature, 1=True
decay_factor = 0.99997592083

vocab_size = 16440 + 1

# check point
checkpoint_path = 'san_cnn_att'

# misc
gpu_id = 0
max_itr = 75001
n_epochs = 200
max_words_q = 22
num_answer = 2000

###################


class SAN_LSTM(tf.keras.Model):
    pass


def map_func(img_name, q, a):
    img_tensor = np.load(img_name.decode('utf-8'))
    return img_tensor, q, a


def get_data():
    dataset = json.load(open(train_input_json, 'r'))
    img_name_train = []
    que_train = []
    ans_train = []
    for item in dataset:
        img_name_train.append(item['img'])
        que_train.append(np.load(item['que']))
        ans_train.append(item['ans'])
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train,
                                                  que_train,
                                                  ans_train))
    dataset = dataset.map(lambda item1, item2, item3:
                          tf.numpy_function(map_func, [item1, item2, item3],
                                            [tf.float32, tf.int32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def train():
    print("loading dataset...")
    dataset = get_data()
    print("vocab size: ")

    print("init model...")
    model = ()




if __name__ == '__main__':
    train()
    # test()
