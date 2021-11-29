FROM ubuntu:hirsute

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin"

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 python3-pip nvidia-cuda-toolkit

# Get file from: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.3.0.98-1+cuda11.5_amd64.deb
COPY libcudnn8_8.3.0.98-1+cuda11.5_amd64.deb /setup/libcudnn8_8.3.0.98-1+cuda11.5_amd64.deb
RUN cd /setup && dpkg -i libcudnn8_8.3.0.98-1+cuda11.5_amd64.deb && cd / && rm -rf /setup

RUN pip3 install torch tqdm tensorboard_logger matplotlib scipy

RUN useradd -m -u 4342 -s /bin/bash ai

USER ai
WORKDIR /home/ai
