#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

DEVICE ?= GPU
MODEL ?=  yolov8n.pt
INPUT ?= "./videos/streat.mp4"

#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=safety
export DOCKER_BUILDKIT=1
MODEL_SIZE ?= n
IMAGE_SIZE ?= 640
MODEL_NAME ?= yolov8${MODEL_SIZE}.pt


DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr   \
	--rm --privileged --pid=host --cap-add=ALL \
    -v /dev/cpu:/dev/cpu \
    -v /sys/devices/system:/sys/devices/system \
	-p 5000:5000 \
	-v ${CURRENT_DIR}:/workspace \
	-w /workspace \
	 ${DOCKER_IMAGE_NAME}

#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: run
.PHONY:  models

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build --rm . -t ${DOCKER_IMAGE_NAME} && \
	docker run ${DOCKER_RUN_PARAMS} \
		bash -c 'cd ./utils/  && python3 ./setup.py build_ext --quiet --inplace'


run: build
	@$(call msg, Running the yolov8 demo ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c '\
				python3 ./app.py '
				
bash: build
	@docker run ${DOCKER_RUN_PARAMS} bash 
	
#----------------------------------------------------------------------------------------------------------------------
# helper functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

