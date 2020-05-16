# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import json
from model import *
import os


###################
# global config  #
##################

print("Loading config")
# data input
train_input_json = "./data/train_data.json"
test_input_json = "./data/test_data.json"

# train params
lr = 0.0003
# lr_decay_start = -1             # when begin to decay lr(-1 never)
batch_size = 128
buffer_size = 1000
embedding_size = 1024     # encoding size of each token in vocab
# rnn_size = 256                  # node# each rnn layer
# rnn_layer = 2
# dim_image = 512
hidden_dim = 1024               # size of common embedding vector
dim_attention = 512             # size of attention embedding
num_output = 2000               # output answer number
# img_norm = 1                    # whether norm image feature, 1=True
# decay_factor = 0.99997592083

vocab_size = 16440 + 1

# check point
experiment_name = 'san_lstm'

# misc
# gpu_id = 0
# max_itr = 75001
n_epochs = 200
max_words_q = 22
num_answer = 2000

###################


def map_func(img_name, q, a):
    img_tensor = np.load(img_name.decode('utf-8'))
    return img_tensor, q, a


def get_data(batch_size=batch_size):
    dataset = json.load(open(train_input_json, 'r'))
    img_name_train = []
    que_train = []
    ans_train = []
    for item in dataset:
        img_name_train.append(item['img'])
        que_train.append(np.load(item['que']))
        ans_train.append(item['ans'])
    num_steps = img_name_train // batch_size
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train,
                                                  que_train,
                                                  ans_train))
    dataset = dataset.map(lambda item1, item2, item3:
                          tf.numpy_function(map_func, [item1, item2, item3],
                                            [tf.float32, tf.int32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, num_steps


def loss_function(real, pred, loss_obj):
    loss_ = loss_obj(real, pred)
    return tf.return_mean(loss_)


def train():
    print("loading dataset...")
    dataset, num_steps = get_data()
    print("vocab size: ")

    print("init model...")
    model = SAN_LSTM(embedding_dim=embedding_size,
                     units=hidden_dim,
                     vocab_size=vocab_size,
                     num_answer=num_output,
                     dim_att=dim_attention)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction='none')

    checkpoint_path = os.path.join("./checkpoints/train/",
                                   experiment_name)
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(chekckpoint=ckpt,
                                              direcotory=checkpoint_path,
                                              max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.lastest_checkpoint:
        start_epoch = int(ckpt_manager.lastest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.lastest_checkpoint)

    loss_plot = []

    EPOCHS = n_epochs

    print("Training Begin...")
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0.0

        for (batch, (img_tensor, que_tensor, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                prediction = model(que_tensor, img_tensor)
                loss = loss_function(target, prediction, loss_object)

            total_loss += loss

            trainable_variables = model.trainable_variables

            gradients = tape.gradient(loss, trainable_variables)

            optimizer.apply_gradients(zip(gradients, trainable_variables))

            if batch % 100 == 0:
                print('Epoch {} Batch {} {} Loss {:.4f}'.format(
                        epoch + 1, batch, loss))
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    train()
    # test()
