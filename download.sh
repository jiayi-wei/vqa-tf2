#!/bin/bash

echo "Download questions"
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -P zip/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P zip/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip -P zip/

echo "Download answers"
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -P zip/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P zip/

ecoh "Unzip annoations"
unzip zip/v2_Questions_Train_mscoco.zip -d annotations/
unzip zip/v2_Questions_Val_mscoco.zip -d annotations/
unzip zip/v2_Questions_Test_mscoco.zip -d annotations/
unzip zip/v2_Annotations_Train_mscoco.zip -d annotations/
unzip zip/v2_Annotations_Val_mscoco.zip -d annotations/

echo "All Done"
