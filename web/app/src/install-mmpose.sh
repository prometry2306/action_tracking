#!/bin/bash
# mmpose のインストール
git clone https://github.com/open-mmlab/mmpose.git
# PyTorch のインストール
pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# MMEngine | MMCV | MMDetection のインストール
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
# Extended COCO API (xtcocotools) のインストール
pip install git+https://github.com/jin-s13/xtcocoapi
# MMPose の依存ライブラリのインストール
cd mmpose && pip install -r requirements.txt && pip install -v -e .
