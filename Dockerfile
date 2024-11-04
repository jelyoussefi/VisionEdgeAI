FROM intel/oneapi

ARG DEBIAN_FRONTEND=noninteractive
USER root

RUN apt  update  -y 
RUN apt  upgrade  -y 

RUN apt install -y  build-essential wget gpg \
					python3-pip python3-dev python3-opencv \
					libopencv-dev libqt5widgets5 pciutils pcm

RUN pip3 install Flask  Flask-Bootstrap

RUN pip3 install --pre -U openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

RUN	pip3 install dpctl dpnp nncf

RUN pip3 install  fire psutil cython ultralytics pyrealsense2



