#conda create --name CFPN python=3.7
#conda activate CFPN

sudo apt-get update
sudo apt-get install libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev -y
conda install pytorch==1.4.0
pip install git+git://github.com/Shamazo/detectron2
conda install opencv wget unzip -y

pip install opencv-python pytorch_msssim range_coder
pip install -e 'git+https://github.com/ahundt/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'
pip install numpy==1.17
