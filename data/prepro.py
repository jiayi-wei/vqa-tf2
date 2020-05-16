import os
import json
import argparse


def main(params):

    train = []
    test = []
    imdir = 'img/%s/COCO_%s_%012d.jpg'

    if params['small'] == 1:
        print('Loading annotations and questions')
        os.system("sh ./download_small.sh")
        anno = json.load(open('annotations/v2_mscoco_val2014_annotations.json', 'r'))

        ques = json.load(open('annotations/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))

        subtype = 'val2014'
        len_ = len(anno['annotations'])

        for i in range(len_):
            ans = anno['annotations'][i]['multiple_choice_answer']
            question_id = anno['annotations'][i]['question_id']
            image_path = imdir % (subtype, subtype, anno['annotations'][i]['image_id'])

            question = ques['questions'][i]['question']

            if i < len_*0.8:
                train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'answer': ans})
            else:
                test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'answer': ans})
        print('Training sample %d, Testing sample %d...' %(len(train), len(test)))

    elif params['small'] == 2:
        print('Loading annotations and questions..')
        os.system("sh ./download_all.sh")
        train_anno = json.load(open('annotations/v2_mscoco_train2014_annotations.json', 'r'))
        val_anno = json.load(open('annotations/v2_mscoco_val2014_annotations.json', 'r'))

        train_ques = json.load(open('annotations/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
        val_ques = json.load(open('annotations/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))
        test_ques = json.load(open('annotations/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))
                
        subtype = 'train2014'
        for i in range(len(train_anno['annotations'])):
            ans = train_anno['annotations'][i]['multiple_choice_answer']
            question_id = train_anno['annotations'][i]['question_id']
            image_path = imdir%(subtype, subtype, train_anno['annotations'][i]['image_id'])

            question = train_ques['questions'][i]['question']

            train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'answer': ans})

        subtype = 'val2014'
        for i in range(len(val_anno['annotations'])):
            ans = val_anno['annotations'][i]['multiple_choice_answer']
            question_id = val_anno['annotations'][i]['question_id']
            image_path = imdir%(subtype, subtype, val_anno['annotations'][i]['image_id'])

            question = val_ques['questions'][i]['question']

            train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'answer': ans})
                
        subtype = 'test2015'
        for i in range(len(test_ques['questions'])):
            question_id = test_ques['questions'][i]['question_id']
            image_path = imdir%(subtype, subtype, test_ques['questions'][i]['image_id'])

            question = test_ques['questions'][i]['question']

            test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'answer': ans})

        print('Training sample %d, Testing sample %d...' %(len(train), len(test)))
    else:
        print("wrong small choice")

    json.dump(train, open('vqa_raw_train.json', 'w'))
    json.dump(test, open('vqa_raw_test.json', 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--small", required=True, type=int, help="1:only val for both train and test; 2: train on train and val test on test")
    
    args = parser.parse_args()
    params = vars(args)
    main(params)

    os.system("rm -r annotations")
