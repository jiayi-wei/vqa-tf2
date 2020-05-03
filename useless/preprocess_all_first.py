# -*- coding: utf-8 -*-

import random
import numpy as np
import json
import argparse

import h5py
import re
import nltk


def get_top_answers(data, params):
    mem = {}
    for item in data:
        ans = item['answer']
        mem[ans] = mem.get(ans, 0) + 1

    ans_count = sorted([(count, a) for a, count in mem.items()], reverse=True)
    print("total answer classes: {}".format(len(ans_count)))

    print("top n popular answer:")
    top_n = []
    unused_ans = 0
    for i in range(len(ans_count)):
        if i < params['num_ans']:
            top_n.append(ans_count[i][1])
        else:
            unused_ans += ans_count[i][0]
        if i < 20:
            print(ans_count[i])

    print("total ans#: {}, unused_ans#: {}".format(len(data), unused_ans))

    return top_n


def filter_data(data, ans_to_idx):
    new_data = []
    for i, item in enumerate(data):
        if item['answer'] in ans_to_idx.keys():
            new_data.append(item)
    print("training sampel reduce from {} to {}".format(len(data), len(new_data)))
    return new_data


def tokenize(s):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])",
                                s) if i != '' and i != ' ' and i != '\n']


def preprocess_question(data, params):
    for i, item in enumerate(data):
        q = item['question']
        if params['token_method'] == 'nltk':
            token = nltk.word_tokenize(str(q).lower())
        else:
            token = tokenize(q)
        item['question_token'] = token
        if i < 10:
            print(token)
        if i % 10000 == 0:
            print("processing {}/{} ({}% done)".format(i, len(data), i*100.0/len(data)))
    return data


def build_vocab_question(data, params):
    word_count_threshold = params['word_count_threshold']

    word_counts = {}
    total_counts = 0
    for item in data:
        for w in item['question_token']:
            word_counts[w] = word_counts.get(w, 0) + 1
            total_counts += 1

    # bad: not enough, good: enough
    bad_words = [w for w, n in word_counts.items() if n <= word_count_threshold]
    vocab = [w for w, n in word_counts.items() if n > word_count_threshold]
    bad_count = sum(word_counts[w] for w in bad_words)
    print("Total words: ", total_counts)
    print("#words in question vocabulary: ", len(vocab))
    print("#bad words: {}/{} ({}%)".format(len(bad_words), len(word_counts), len(bad_words)*100.0/len(word_counts)))
    print("#UNKS: {}/{} ({}%)".format(bad_count, total_counts, bad_count*100.0/total_counts))

    # substitude bad words with UNK
    vocab.append('UNK')

    for item in data:
        token = item['question_token']
        final_token = []
        for w in token:
            if w in word_counts.keys() and word_counts[w] > word_count_threshold:
                final_token.append(w)
            else:
                final_token.append('UNK')
        item['final_question_token'] = final_token

    return data, vocab


def apply_vocab_question(test_data, word_to_idx):
    for item in test_data:
        token = item['question_token']
        final_token = [w if w in word_to_idx.keys() else 'UNK' for w in token]
        item['final_question_token'] = final_token
    return test_data


def get_unique_img(data):
    count_img = {}
    N = len(data)
    img_pos = np.zeros(N, dtype='uint32')
    for item in data:
        if item['img_path'] in count_img.keys():
            count_img[item['img_path']] += 1
        else:
            count_img[item['img_path']] = 1
    unique_img = [p for p, c in count_img.items()]
    img_to_idx = {p: i+1 for i, p in enumerate(unique_img)}

    for i, item in enumerate(data):
        img_pos[i] = img_to_idx[item['img_path']]

    return unique_img, img_pos


def encode_question(data, params, word_to_idx):
    max_length = params['max_length']
    N = len(data)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    # question_counter = 0

    for i, item in enumerate(data):
        question_id[i] = item['ques_id']
        label_length[i] = min(max_length, len(item['final_question_token']))
        for k, w in enumerate(item['final_question_token']):
            if k < max_length:
                label_arrays[i, k] = word_to_idx[w]

    return label_arrays, label_length, question_id


def max_length(data):
    max_len = max(len(item['final_question_token']) for item in data)
    return max_len


def encode_answer(data, ans_to_idx):
    N = len(data)
    ans_arrays = np.zeros(N, dtype='uint32')
    for i, item in enumerate(data):
        ans_arrays[i] = ans_to_idx[item['answer']]
    return ans_arrays


def main(params):
    trains = json.load(open(params['input_train_json'], 'r'))
    tests = json.load(open(params['input_test_json'], 'r'))

    # get top n answers
    top_ans = get_top_answers(trains, params)
    ans_to_idx = {w: i+1 for i, w in enumerate(top_ans)}
    idx_to_ans = {i+1: w for i, w in enumerate(top_ans)}
    json.dump(idx_to_ans, open('./data/idx_to_ans.json', 'w'))

    # filter data whose answer is not in top_ans
    trains = filter_data(trains, ans_to_idx)

    # shuffle train samples order
    random.seed(123)
    random.shuffle(trains)

    # tokenize data
    trains = preprocess_question(trains, params)
    tests = preprocess_question(tests, params)

    # creat the question vocabulary
    trains, vocab = build_vocab_question(trains, params)
    idx_to_word = {i+1: w for i, w in enumerate(vocab)}
    word_to_idx = {w: i+1 for i, w in enumerate(vocab)}
    json.dump(word_to_idx, open('word_to_idx.json', 'w'))

    tests = apply_vocab_question(tests, word_to_idx)

    # train_q is (len(trains), max_length) array
    # train_q_len is (len(trains) arrays, is the length of ith question
    # train_q_id is the (len(trains)) array, is the id the ith question
    train_q, train_q_len, train_q_id = encode_question(trains, params, word_to_idx)
    test_q, test_q_len, test_q_id = encode_question(tests, params, word_to_idx)

    # get the unique image list for both train and test
    # img_pos has same long as data. like len(img_pos_train) = len(trains)
    # img_pos has the index of the image path in unique_img list
    # So img_pos connect data and unique_img
    unique_img_train, img_pos_train = get_unique_img(trains)
    unique_img_test, img_pos_test = get_unique_img(tests)

    # encode answer
    A = encode_answer(trains, ans_to_idx)

    # save whole data to h5 file
    f = h5py.File(params['output_h5'], 'w')
    f.create_dataset('ques_train', dtype='uint32', data=train_q)
    f.create_dataset('ques_length_train', dtype='uint32', data=train_q_len)
    f.create_dataset('answers', dtype='uint32', data=A)
    f.create_dataset('question_id_train', dtype='uint32', data=train_q_id)
    f.create_dataset('img_pos_train', dtype='uint32', data=img_pos_train)

    f.create_dataset('ques_test', dtype='uint32', data=test_q)
    f.create_dataset('ques_length_test', dtype='uint32', data=test_q_len)
    f.create_dataset('question_id_test', dtype='uint32', data=test_q_id)
    f.create_dataset('img_pos_test', dtype='uint32', data=img_pos_test)

    f.close()
    print("H5 {} write done!".format(params['output_h5']))

    # save output json file
    out = {}
    out['idx_to_word'] = idx_to_word
    out['idx_to_ans'] = idx_to_ans
    out['unique_img_train'] = unique_img_train
    out['unique_img_test'] = unique_img_test
    json.dump(out, open(params['output_json'], 'w'))
    print("Json {} write doen!".format(params['output_json']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # argument input
    parser.add_argument('--input_train_json', default='./data/vqa_raw_train.json')
    parser.add_argument('--input_test_json', default='./data/vqa_raw_test.json')
    parser.add_argument('--num_ans', default=100, type=int)

    parser.add_argument('--output_json', default='./data/prepro.json')
    parser.add_argument('--output_h5', default='./data/prepro.h5')

    parser.add_argument('--max_length', default=25, type=int)
    parser.add_argument('--word_count_threshold', default=0, type=int)
    parser.add_argument('--num_test', default=0, type=int)
    parser.add_argument('--token_method', default='nltk')
    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()

    params = vars(args)
    print(json.dumps(params, indent=2))

    main(params)
