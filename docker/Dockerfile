# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.07-py3
FROM $BASE_IMAGE

# If your are in China
# RUN if [ -f /etc/apt/sources.list.d/cuda.list ]; then \
#     rm /etc/apt/sources.list.d/cuda.list; \
#     fi \
#     && if [ -f /etc/apt/sources.list.d/nvidia-ml.list ]; then \
#     rm /etc/apt/sources.list.d/nvidia-ml.list; \
#     fi && \
#     sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
#     sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
#     pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update -yq --fix-missing \
    && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    git

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# additional libraries
RUN pip install ninja imageio imageio-ffmpeg xatlas gdown

# install nvdiffrast
RUN git config --global http.proxy socks://219.223.185.245:1080 && pip install git+https://github.com/NVlabs/nvdiffrast/

# HDR image support
# RUN http_proxy='socks://219.223.185.245:1080' https_proxy='socks://219.223.185.245:1080' imageio_download_bin freeimage
