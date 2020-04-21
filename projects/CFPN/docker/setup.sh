# docker run --rm --mount type=bind,source="$(pwd)"/CFPN,target=/workspace/CFPN -it detectron2-compressive bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
mkdir CFPN/datasets/
tar -xf VOCtrainval_11-May-2012.tar -C CFPN/datasets/
mv CFPN/datasets/VOCdevkit/VOC2012 CFPN/datasets/
mkdir CFPN/datasets/coco/
unzip val2017.zip -d CFPN/datasets/coco
