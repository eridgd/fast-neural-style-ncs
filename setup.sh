#! /bin/bash

mkdir models
cd models
wget -c -O kanagawa.zip https://www.dropbox.com/s/a17fvuur9kms49y/kanagawa.zip?dl=1
unzip kanagawa.zip

## Uncomment if training a model from scratch
#mkdir data
#cd data
#wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
#mkdir bin
#wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
#unzip train2014.zip
