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
learning_rate = 0.01
# lr_decay_start = -1             # when begin to decay lr(-1 never)
batch_size = 2048
buffer_size = 1000
embedding_size = 1024     # encoding size of each token in vocab
# rnn_size = 256                  # node# each rnn layer
# rnn_layer = 2
# dim_image = 512
hidden_dim = 1024               # size of common embedding vector
dim_attention = 512             # size of attention embedding
num_output = 2000               # output answer number
# img_norm = 1                    # whether norm image feature, 1=True
decay_factor = 0.99997592083

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
    num_steps = len(img_name_train) // batch_size
    print("Total train samples {}, batch size {}, steps for each epoch {}".format(
            len(img_name_train), batch_size, num_steps))
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

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000,
        end_learning_rate=0.00001,
        power=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    checkpoint_path = os.path.join("./checkpoints/",
                                   experiment_name)
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                              directory=checkpoint_path,
                                              max_to_keep=5)
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    loss_plot = []

    EPOCHS = n_epochs

    print("Training Begin...")
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0.0

        for (batch, (img_tensor, que_tensor, target)) in enumerate(dataset):
            with tf.GradientTape() as tape:
                prediction, layer_1_w, layer_2_w = model(que_tensor,
                                                         img_tensor)
                loss = loss_object(target, prediction)

            total_loss += loss.numpy()
            if tf.reduce_any(tf.math.is_nan(loss)):
                print(loss.numpy())
                print(target.numpy())
                print(prediction.numpy())
                quit()

            trainable_variables = model.trainable_variables

            gradients = tape.gradient(loss, trainable_variables)
            gradients = [(tf.clip_by_value(grad, -10.0, 10.0)) for grad in gradients]

            optimizer.apply_gradients(zip(gradients, trainable_variables))

            if batch % 50 == 0:
                print('Epoch: {} Batch: {} Loss: {:.4f} LR: {:.6f}'.format(
                        epoch + 1, batch, loss.numpy(),
                        optimizer._decayed_lr(tf.float32)))

        loss_plot.append(total_loss / num_steps)

        ckpt_manager.save()

        print('Epoch {} average loss is {:.6f}'.format(epoch + 1,
                                                       total_loss / num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    np.save(os.path.join(checkpoint_path, "loss.npy"), loss_plot)
    print("Training Done!")


if __name__ == '__main__':
    train()
    # test()
