# -*- coding: utf-8 -*-

import random
import numpy as np
import json
import argparse
import tensorflow as tf
import re
import os
from tqdm import tqdm


def get_top_answers(data, params):
    mem = {}
    for item in data:
        ans = item['answer']
        mem[ans] = mem.get(ans, 0) + 1
    ans_count = sorted([(count, a) for a, count in mem.items()], reverse=True)
    top_n = []
    unused_ans = 0
    for i in range(len(ans_count)):
        if i < params['num_ans']:
            top_n.append(ans_count[i][1])
        else:
            unused_ans += ans_count[i][0]
    print("total ans#: {}, unused ans#: {}".format(len(data), unused_ans))
    return top_n


def filter_data(data, ans_to_idx):
    new_data = []
    for i, item in enumerate(data):
        if item['answer'] in ans_to_idx.keys():
            new_data.append(item)
    print("training samples reduce from {} to {}".format(len(data), len(new_data)))
    return new_data


def tokenize1(s):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                s) if i != '' and i != ' ' and i != '\n']


def tokenize2(s):
    return [i for i in re.split(r"([-.\",:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                s) if i != '' and i != ' ' and i != '\n']


def words_stat(data):
    stat1 = []
    stat2 = []
    for i, item in enumerate(data):
        q = item['question']
        token1 = tokenize1(q)
        for t in token1:
            stat1.append(t)
        token2 = tokenize2(q)
        for t in token2:
            stat2.append(t)
        if i % 10000 == 0:
            print(i, "/", len(data))
    print("token1_with: ", len(set(stat1)))
    print("token2_without: ", len(set(stat2)))


def max_length(data):
    return max(len(t) for t in data)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img, image_path


def feature_path(p, cate, train_flag):
    # ./data/./img/train2014/COCO_train2014_000000178619.jpg
    feature_root = './data/features/'
    if not os.path.exists(feature_root):
        os.mkdir(feature_root)
    name = p.split('/')[-1].split('.')[0] + '.npy'
    directory = os.path.join(feature_root, cate)
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = os.path.join(directory, 'train' if train_flag else 'test')
    if not os.path.exists(directory):
        os.mkdir(directory)
    return os.path.join(directory, name)


def extract_feature(img_path, train_flag, image_features_extract_model):
    unique_img = list(set(img_path))
    unique_img = list(map(lambda x: os.path.join('./data/', x), unique_img))
    print("total image# to preprocess: ", len(unique_img))

    image_dataset = tf.data.Dataset.from_tensor_slices(unique_img)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(32)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for f, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            path_of_feature = feature_path(path_of_feature, cate='img', train_flag=train_flag)
            np.save(path_of_feature, f.numpy())


def main(params):
    trains = json.load(open(params['input_train_json'], 'r'))
    tests = json.load(open(params['input_test_json'], 'r'))

    # get top n answers
    top_ans = get_top_answers(trains, params)
    ans_to_idx = {w: i+1 for i, w in enumerate(top_ans)}
    idx_to_ans = {i+1: w for i, w in enumerate(top_ans)}
    json.dump(idx_to_ans, open('./data/idx_to_ans.json', 'w'))

    # fiter data whose answer is not in top_ans
    trains = filter_data(trains, ans_to_idx)

    # shuffle train samples order
    random.seed(123)
    random.shuffle(trains)

    # split
    all_train_img_path = []
    all_train_question = []
    all_train_answer = []

    all_test_img_path = []
    all_test_question = []

    for item in trains:
        all_train_img_path.append(item['img_path'])
        all_train_question.append(item['question'])
        all_train_answer.append(ans_to_idx[item['answer']])

    for item in tests:
        all_test_img_path.append(item['img_path'])
        all_test_question.append(item['question'])

    # get top_k words in train
    # words_stat(trains)  # 16442 words

    # tokenize using tf
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=params['top_k'],
                                                      oov_token='<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(all_train_question)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    word_idx = tokenizer.get_config()
    json.dump(word_idx, open('./data/word_idx.json', 'w'))

    all_train_question_seq = tokenizer.texts_to_sequences(all_train_question)
    all_test_question_seq = tokenizer.texts_to_sequences(all_test_question)

    maxLen = max(max_length(all_train_question_seq),
                 max_length(all_test_question_seq))
    all_train_question_vec = tf.keras.preprocessing.sequence.pad_sequences(all_train_question_seq,
                                                                           maxlen=maxLen,
                                                                           padding='pre')
    all_test_question_vec = tf.keras.preprocessing.sequence.pad_sequences(all_test_question_seq,
                                                                          maxlen=maxLen,
                                                                          padding='pre')
    # print(type(all_test_question_vec), all_test_question_vec.shape, all_test_question_vec[0])
    print(all_test_question_vec.shape, all_train_question_vec.shape, type(all_train_question_vec[0]))
    print('tokenizing done!')

    # using vgg19 pool5 to extract image feature
    # image_model = tf.keras.applications.VGG19(include_top=False,
    #                                          weights='imagenet')
    # new_input = image_model.input
    # hidden_layer = image_model.layers[-1].output
    # image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    print("image model ready")

    # extract image features
    # Train
    # extract_feature(all_train_img_path, True, image_features_extract_model)
    print("train image done.")

    # Test
    # extract_feature(all_test_img_path, False, image_features_extract_model)
    print("test image done.")

    # save tokenized question
    train_json = []
    test_json = []

    for i in range(len(all_train_img_path)):
        ques_feature = all_train_question_vec[i]
        ques_feature_path = feature_path(str(i+1), cate='que', train_flag=True)
        np.save(ques_feature_path, ques_feature)
        dict_ = {'img': feature_path(all_train_img_path[i], cate='img', train_flag=True),
                 'que': ques_feature_path,
                 'ans': all_train_answer[i]}
        train_json.append(dict_)
        if (i+1) % 10000 == 0:
            print("train write {}%".format((i+1)*100.0 / len(all_train_img_path)))
    json.dump(train_json, open(params['output_train_json'], 'w'))
    print('train write done')

    for i in range(len(all_test_img_path)):
        ques_feature = all_test_question_vec[i]
        ques_feature_path = feature_path(str(i+1), cate='que', train_flag=False)
        np.save(ques_feature_path, ques_feature)
        dict_ = {'img': feature_path(all_train_img_path[i], cate='img', train_flag=False),
                 'que': ques_feature_path}
        test_json.append(dict_)
        if (i+1) % 10000 == 0:
            print("test write {}%".format((i+1)*100.0 / len(all_test_img_path)))
    json.dump(test_json, open(params['output_test_json'], 'w'))
    print('test write done')

    print("All Preprocess Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # argument input
    parser.add_argument('--input_train_json', default='./data/vqa_raw_train.json')
    parser.add_argument('--input_test_json', default='./data/vqa_raw_test.json')
    parser.add_argument('--num_ans', default=2000, type=int)

    parser.add_argument('--top_k', default=16440, type=int, help='16442 is all words in train+val ques')

    parser.add_argument('--output_train_json', default='./data/train_data.json')
    parser.add_argument('--output_test_json', default='./data/test_data.json')

    args = parser.parse_args()
    params = vars(args)

    main(params)
