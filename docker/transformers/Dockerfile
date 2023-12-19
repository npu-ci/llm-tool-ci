FROM zhangsibo1129/ubuntu-cann-torch21-py39:latest

ARG REF=main
WORKDIR /root

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout $REF

ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd transformers && \
    pip install --no-cache-dir -e ".[dev-torch]" -i ${PIP_SOURCE_URL}

# Avoid import error caused by scikit-learn
ENV LD_PRELOAD=/root/miniconda/envs/torch_npu/lib/python.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
