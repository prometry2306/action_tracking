# Ubuntu22.04 + CUDA11.8 のGPUコンテナ
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="mshinoda"

RUN apt-get update && apt-get dist-upgrade -y
RUN apt-get -y install apt-utils
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV TZ=Asia/Tokyo
ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_ROOTDIR /opt/app
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get install -y apt-file software-properties-common

# coreutils for cat /meld for diff
RUN apt-get install -y coreutils vim less nkf jq zip unzip wget meld sudo git curl telnet locate
RUN apt-get install -y python3-ldb-dev gcc libcurl4-openssl-dev
RUN apt-get install -y tmux
# for pip install
RUN apt-get install -y python3-distutils python3-testresources
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN echo 'alias ll="ls -al"' >> /root/.bashrc

# install application materials
USER root
WORKDIR ${PROJECT_ROOTDIR}/src

# mmpose のインストール
RUN git clone https://github.com/open-mmlab/mmpose.git
# PyTorch のインストール
RUN pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# MMEngine | MMCV | MMDetection のインストール
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0"
RUN mim install "mmdet>=3.0.0"
# Extended COCO API (xtcocotools) のインストール
RUN pip install git+https://github.com/jin-s13/xtcocoapi
# MMPose の依存ライブラリのインストール
RUN cd mmpose && pip install -r requirements.txt && pip install -v -e .

# python ライブラリのインストール
WORKDIR ${PROJECT_ROOTDIR}
COPY requirements.txt ${PROJECT_ROOTDIR}/requirements.txt
RUN pip3 install --trusted-host pypi.python.org --no-cache-dir -r ${PROJECT_ROOTDIR}/requirements.txt
WORKDIR ${PROJECT_ROOTDIR}/src
