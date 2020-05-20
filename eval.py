# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import json
from model_2lstm import *
import os


#######################
#    Global Config    #

print("Loading config")
test_input_json = "./data/test_data.json"
train_input_json = "./data/train_data.json"
with_target = True

batch_size = 1024
embedding_size = 1024
hidden_dim = 1024
dim_attention = 512
num_output = 2000 + 1
vocab_size = 16440 + 1

experimental_name = "san_2lstm_all"

#######################


def map_func_no_target(img_name, q):
    img_tensor = np.load(img_name.decode('utf-8'))
    q = np.load(q)
    a = -1
    return img_tensor, q, a, img_name


def map_func(img_name, q, a):
    img_tensor = np.load(img_name.decode('utf-8'))
    q = np.load(q)
    return img_tensor, q, a, img_name


def get_data(json_file,
             batch_size=batch_size,
             with_target=True):
    dataset = json.load(open(json_file, 'r'))
    img_name_test = []
    que_test = []
    if with_target:
        ans_test = []
    for item in dataset:
        img_name_test.append(item['img'])
        que_test.append(item['que'])
        if with_target:
            ans_test.append(item['ans'])
    num_steps = len(img_name_test) // batch_size
    print("Total test samples {}, batch_size {}, steps needed {}".format(
            len(img_name_test), batch_size, num_steps))
    dataset = tf.data.Dataset.from_tensor_slice((img_name_test,
                                                 que_test,
                                                 ans_test))
    if with_target:
        dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(map_func,
                                                                            [item1, item2, item3]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func_no_target,
                                                                     [item1, item2]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch_size(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def test():
    print("Loading dataset...")
    if with_target:
        dataset = get_data(json_file=train_input_json,
                           with_target=with_target)
    else:
        dataset = get_data(json_file=test_input_json,
                           with_target=with_target)

    print("Init model...")
    model = SAN_LSTM(embedding_dim=embedding_size,
                     units=hidden_dim,
                     vocab_size=vocab_size,
                     num_answer=num_output,
                     dim_att=dim_attention,
                     training=False)

    checkpoint_path = os.path.join("./checkpoints",
                                   experimental_name)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    model.loat_weights(latest)

    correct, total = 0, 0
    res = []
    idx_to_ans = json.load("./data/idx_to_ans.json")
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json("./data/word_idx.json")

    extention = "train" if with_target else "test"
    save_path = os.path.join(checkpoint_path, extention)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    w1_path = os.path.join(save_path, 'w1')
    w2_path = os.path.join(save_path, 'w2')
    res_path = os.path.join(save_path, 'res.json')

    img_path_sub = "./data/img/{}/"
    for (batch, (img_tensor, que_tensor, target, img_path)) in enumerate(dataset):
        prediction, layer_1_w, layer_2_w = model(que_tensor,
                                                 img_tensor)
        prediction = tf.math.argmax(prediction, axis=1)

        for i in range(prediction.shape[0]):
            if with_target:
                total += 1
                if prediction[i] == target[i]:
                    correct += 1
            name = "{:04d}_{:04d}_".format(batch, i)
            w1 = layer_1_w[i].numpy()
            w1_path_here = os.path.join(w1_path, name+"w1.npy")
            np.load(w1_path_here, w1)
            w2 = layer_2_w[i].numpy()
            w2_path_here = os.path.join(w2_path, name+"w2.npy")
            np.load(w2_path_here, w2)
            que = que_tensor[i]
            que = tokenizer.sequences_to_texts(que)
            fig_path = str(img_path[i].numpy())
            if "train2014" in fig_path:
                cat = "train2014"
            elif "val2014" in fig_path:
                cat = "val2014"
            elif "test2015" in fig_path:
                cat = "val2015"
            else:
                print(fig_path)
                quit()
            ext = fig_path.split("/")[-1]
            ext = ext.split(".")[0]
            fig_path = img_path_sub.format(cat) + ext + ".jpg"
            ans = target[i].numpy()
            if with_target:
                ans = idx_to_ans[ans]
            else:
                ans = "None"
            pred = prediction[i].numpy()
            pred = idx_to_ans[pred]
            dic_ = {"w1": w1_path_here,
                    "w2": w2_path_here,
                    "que": que,
                    "ans": ans,
                    "pre": pred,
                    "img": fig_path}
            res.append(dic_)
    print("Accu {}".format(correct / total))
    json.dump(res_path, res)


if __name__ == "__main__":
    test()
