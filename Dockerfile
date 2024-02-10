FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    curl \
    python3.9 \
    python3.9-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pythonコマンドを通すための設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --set python /usr/bin/python3.9
RUN python -m pip install --upgrade pip

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python -

# Poetryのパスの設定
ENV PATH /root/.local/bin:$PATH

# Poetryが仮想環境を生成しないようにする
RUN poetry config virtualenvs.create false

# ↑の設定を行った際、Poetryのパッケージインストール先が
# /usr/lib/python3.9/site-packagesになっているため
# それをPythonに認識させる必要がある
ENV PYTHONPATH /usr/lib/python3.9/site-packages

WORKDIR /work
