FROM openvino/ubuntu22_dev

ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt  update  -y --allow-insecure-repositories

RUN apt install -y  build-essential wget gpg \
					python3-pip python3-dev python3-opencv \
					libopencv-dev libqt5widgets5 pciutils

RUN	pip3 install dpctl dpnp nncf
RUN pip3 install Flask Flask-RESTful Flask-Bootstrap
RUN pip3 install  fire psutil cython ultralytics pyrealsense2

RUN apt install -y pcm
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

