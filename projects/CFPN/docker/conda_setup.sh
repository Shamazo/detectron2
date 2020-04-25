#conda create --name CFPN python=3.6
conda activate CFPN

sudo apt-get update
sudo apt-get install libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev -y
conda install numpy==1.17 opencv pytorch wget unzip
pip install pytorch_msssim 

pip install git+git://github.com/Shamazo/detectron2
pip install -e 'git+https://github.com/ahundt/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'


