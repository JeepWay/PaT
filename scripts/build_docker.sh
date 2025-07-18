#!/bin/bash

CPU_PARENT=pytorch/pytorch
GPU_PARENT=pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

TAG=jeepway/pat
VERSION=$(cat ./pat/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  TAG="${TAG}-gpu"
else
  PARENT=${CPU_PARENT}
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} -t ${TAG}:${VERSION} .
docker tag ${TAG}:${VERSION} ${TAG}:latest

# if [[ ${RELEASE} == "True" ]]; then
#   docker push ${TAG}:${VERSION}
#   docker push ${TAG}:latest
# fi