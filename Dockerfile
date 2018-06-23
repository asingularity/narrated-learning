FROM nvidia/cuda:8.0-devel
MAINTAINER asingularity <csaba.petre@gmail.com>

RUN apt-get update && apt-get install -y gedit
RUN apt-get install -qqy x11-apps

RUN apt-get -y install python3-numpy
RUN apt-get -y install cython3
RUN apt-get install -y python3-pip
RUN pip3 install pycuda

# then github scikit-cuda (this works!)

RUN pip3 install opencv-python
RUN apt-get -y install python3-matplotlib

ENV DISPLAY :0

