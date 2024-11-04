FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
USER root

RUN apt update -y && \
    apt install -y build-essential wget gpg curl \
    python3-pip python3-dev python3-opencv \
    libopencv-dev libqt5widgets5 pciutils pcm


RUN pip3 install Flask nncf fire psutil cython ultralytics  --break-system-packages
RUN pip3 install --pre -U openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --break-system-packages
RUN pip3 install flask_bootstrap --break-system-packages

WORKDIR /opt/intel
RUN curl -L https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.4/linux/l_openvino_toolkit_ubuntu24_2024.4.0.16579.c3152d32c9c_x86_64.tgz \
    --output openvino_2024.4.0.tgz
RUN tar -xf openvino_2024.4.0.tgz
RUN rm -f openvino_2024.4.0.tgz
RUN mv l_openvino_toolkit_ubuntu24_2024.4.0.16579.c3152d32c9c_x86_64 /opt/intel/openvino_2024.4.0
RUN ln -sf /opt/intel/openvino_2024.4.0 /opt/intel/openvino
RUN cd /opt/intel/openvino/ && \
    ./install_dependencies/install_openvino_dependencies.sh -y && \
    ./samples/cpp/build_samples.sh

RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  		gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

RUN echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
		tee /etc/apt/sources.list.d/intel-gpu-noble.list

RUN apt update -y 
RUN apt install -y libze1 intel-level-zero-gpu intel-opencl-icd clinfo

WORKDIR /tmp
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.8.0/intel-driver-compiler-npu_1.8.0.20240916-10885588273_ubuntu24.04_amd64.deb
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.8.0/intel-fw-npu_1.8.0.20240916-10885588273_ubuntu24.04_amd64.deb
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.8.0/intel-level-zero-npu_1.8.0.20240916-10885588273_ubuntu24.04_amd64.deb
RUN dpkg -i *.deb
RUN rm *.deb

ENV PATH=/root/openvino_cpp_samples_build/intel64/Release/:${PATH}
