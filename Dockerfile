FROM ubuntu:24.10

ARG DEBIAN_FRONTEND=noninteractive
USER root

RUN apt update -y && \
    apt install -y build-essential wget gpg curl pciutils git cmake \
    python3-pip python3-dev python3-setuptools \
    python3-opencv libopencv-dev 

RUN pip3 install Flask flask_bootstrap nncf fire psutil cython \
                 ultralytics openvino-dev[onnx] --break-system-packages

#-------------- Graphic Drivers

RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  		gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

RUN echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
		tee /etc/apt/sources.list.d/intel-gpu-noble.list

RUN apt update -y 
RUN apt install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo

#-------------- NPU Driver

WORKDIR /tmp

RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-driver-compiler-npu_1.10.0.20241107-11729849322_ubuntu24.04_amd64.deb
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-fw-npu_1.10.0.20241107-11729849322_ubuntu24.04_amd64.deb
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-level-zero-npu_1.10.0.20241107-11729849322_ubuntu24.04_amd64.deb
RUN dpkg -i *.deb
RUN rm *.deb

#-------------- PCM 

RUN git clone --recursive https://github.com/intel/pcm
RUN cd ./pcm && mkdir build && cd build && \
	cmake .. && cmake --build . --parallel && make install && \
	rm -rf /tmp/pcm
	
#-------------- Models
	
COPY ./utils/models.sh /tmp/
WORKDIR /opt/models
RUN /tmp/models.sh && rm -rf  /tmp/models.sh
    
WORKDIR /workspace
