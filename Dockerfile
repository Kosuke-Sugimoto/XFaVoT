FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    python3 \
    python3-pip \
    python-is-python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.0+cu117 \
    torchvision==0.15.1+cu117 \
    torchaudio==2.0.1 \
    && pip install \
    librosa \
    scipy \
    soxr \
    tqdm \
    wandb \
    tensorboard \
    omegaconf \
    munch

WORKDIR /work
