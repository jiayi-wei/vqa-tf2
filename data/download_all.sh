#!/bin/bash

echo "Download questions"
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P zip/

echo "Download answers"
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/

echo "Unzip annoations"
unzip zip/v2_Questions_Train_mscoco.zip -d annotations/
unzip zip/v2_Questions_Val_mscoco.zip -d annotations/
unzip zip/v2_Questions_Test_mscoco.zip -d annotations/
unzip zip/v2_Annotations_Train_mscoco.zip -d annotations/
unzip zip/v2_Annotations_Val_mscoco.zip -d annotations/

echo "Download images"
wget http://images.cocodataset.org/zips/train2014.zip -P zip/
wget http://images.cocodataset.org/zips/val2014.zip -P zip/
wget http://images.cocodataset.org/zips/test2015.zip -P zip/

echo "Unzip images"
unzip zip/train2014.zip -d img/
unzip zip/val2014.zip -d img/
unzip zip/test2015.zip -d img

echo "Prepare annotation"
# python ./prepro.py

echo "Remove useless files"
rm -r zip
# rm -r annoations

echo "All Done"
