# Tensorflow 2.0 Implementation of Stacked Attention Networks for Image Question Answering

In this repo, I implement the [SAN] (https://arxiv.org/pdf/1511.02274.pdf) with TensorFlow 2.0 framework. Only the LSTM is utilized to extract text presentation. Only the VQA-v2 dataset is employed in my experiment.

### Requirements
Python 3.7
Tensorflow 2.2.0

### Prepare Data
All necessary VQA-v2 data (image/question/answer) could be downloaded with the scripts in the "data" folder.
```
cd data
```
If you only want to work on a small dataset (train on 0.8*val2014 and test on the left val2014):
```
python prepro.py --small=1
```
If train the model on train2014+val2014 and test on test2015:
```
python prepro.py --small=2
```

### Extract image and question feature
I use the VGG19 model pre-trained on ImageNet to extract image features. The tensorflow built-in tokenizer is utilized to extract question features.
In the configuration, the top-2000 frequent answers are remained for training, the vocabulary size covers all questions words, and the batch size is 64. Custom for your experiment environment.
```
python preprocess.py
```

### Train and Eval
To train the model:
```
python main.py
```
Checkpoints will be saved in the "./checkpoints" folder with the experiment name.

You can change the boolean variable "with_target" to evaluate on train+val or test.
```
python eval.py
```
With 80 epochs of training, my model achieves 49% accuracy for the overall category on test data.
