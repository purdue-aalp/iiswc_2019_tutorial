# Based on Ubuntu 18.04
FROM ubuntu:18.04

# Originially created for the IISWC 2019 Tutorial
MAINTAINER Roland Green <rgreen.dev@gmail.com>

# Dependencies for GPGPU-Sim + some extras
RUN apt-get update -y && apt-get install -y \
    bison \
    build-essential \
    flex \
    g++-5 \
    gcc-5 \
    libglu1-mesa-dev \
    libxi-dev \
    libxmu-dev \
    wget \
    xutils-dev \
    zlib1g-dev \
    git \
    vim \
    libxml2

# Change symlinks to point to GCC 5.
RUN rm /usr/bin/gcc && ln -s /usr/bin/gcc-5 /usr/bin/gcc
RUN rm /usr/bin/g++ && ln -s /usr/bin/g++-5 /usr/bin/g++

# Download CUDA Toolkit version 10.1.
RUN wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run

# Must be silent. Otherwise it requires user interaction
# We only need the toolkit
RUN sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit

# Download GPU Computing SDK version 3.2.16.
RUN wget http://developer.download.nvidia.com/compute/cuda/3_2_prod/sdk/gpucomputingsdk_3.2.16_linux.run

# Install the GPU Computing SDK toolkit.
RUN bash gpucomputingsdk_3.2.16_linux.run

RUN mv /root/NVIDIA_GPU_Computing_SDK/C /usr/local/cuda && \
    mv /root/NVIDIA_GPU_Computing_SDK/shared /usr/local/cuda

# Place the environment setup lines in a file
RUN echo " \
    export CUDAHOME=/usr/local/cuda; \
    export PATH=$PATH:/usr/local/cuda/bin; \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib; \
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/C/lib:/usr/local/cuda/shared/lib; \
    export CUDA_INSTALL_PATH=/usr/local/cuda; \
    export NVIDIA_COMPUTE_SDK_LOCATION=/usr/local/cuda; \
    " >> /root/env

# Source this in the .bashrc file
RUN echo "source /root/env" >> /root/.bashrc

WORKDIR /usr/local/cuda/C/common
RUN make 2> /dev/null
WORKDIR /usr/local/cuda/shared
RUN make 2> /dev/null

WORKDIR /
