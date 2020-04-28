#!/bin/bash
#echo 'downloading and unzipping coco'
#mkdir datasets
#cd datasets
#mkdir coco
#cd coco
#curl 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip' --output 'annotations_trainval2017.zip'
#unzip 'annotations_trainval2017.zip'
#curl 'http://images.cocodataset.org/zips/train2017.zip' --output 'train2017.zip'
#unzip 'train2017.zip'
#cd
#conda activate azureml_py36_pytorch
#echo 'installing packages'
#pip install pytorch-msssim range-coder
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
#cd detectron2
#pip install -e .
pip install range-coder
pip install pytorch-msssim