#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

DEVICE ?= GPU
MODEL ?= yolov8n.pt
INPUT ?= "./videos/streat.mp4"

# Proxy settings (inherited from the host environment if set)
HTTP_PROXY := $(HTTP_PROXY)
HTTPS_PROXY := $(HTTPS_PROXY)
NO_PROXY := $(NO_PROXY)

#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=safety
export DOCKER_BUILDKIT=1
MODEL_SIZE ?= n
IMAGE_SIZE ?= 640
MODEL_NAME ?= yolov8${MODEL_SIZE}.pt

# Docker run parameters with proxy settings
DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr   \
	--privileged -v /dev:/dev \
	-p 5000:5000 \
	-v ${CURRENT_DIR}:/workspace \
	-w /workspace \
	-e http_proxy=${HTTP_PROXY} \
	-e https_proxy=${HTTPS_PROXY} \
	-e no_proxy=${NO_PROXY} \
	-e MKL_THREADING_LAYER=gnu \
	${DOCKER_IMAGE_NAME}

#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: run
.PHONY: models

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build --rm . -t ${DOCKER_IMAGE_NAME} \
		--build-arg http_proxy=${HTTP_PROXY} \
		--build-arg https_proxy=${HTTPS_PROXY} \
		--build-arg no_proxy=${NO_PROXY} && \
	docker run ${DOCKER_RUN_PARAMS} \
		bash -c 'cd ./utils/  && python3 ./setup.py build_ext --quiet --inplace'

run: build
	@$(call msg, Running the yolov8 demo ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c ' \
				python3 ./app.py '

bash: build
	@docker run ${DOCKER_RUN_PARAMS} bash 

#----------------------------------------------------------------------------------------------------------------------
# Helper Functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

