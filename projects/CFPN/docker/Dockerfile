FROM pytorch/pytorch
RUN apt-get update
RUN apt-get install libglib2.0-0 libsm6 libxrender1 libfontconfig1 libxext6 libxrender-dev -y
RUN pip install git+git://github.com/Shamazo/detectron2
RUN conda install opencv -y
RUN pip install opencv-python
RUN pip install pytorch_msssim
RUN pip install -e 'git+https://github.com/ahundt/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'
RUN pip install numpy==1.17
RUN conda install wget -y
RUN conda install unzip -y
COPY setup.sh /workspace
