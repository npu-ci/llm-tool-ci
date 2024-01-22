FROM zhangsibo1129/ubuntu-cann-torch21-py39:latest

ARG REF=main
WORKDIR /root

RUN git clone https://github.com/huggingface/accelerate.git && \
    cd accelerate && \
    git checkout $REF

ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd accelerate && \
    pip install --no-cache-dir -e ".[dev]" -i ${PIP_SOURCE_URL}

# Avoid import error caused by scikit-learn
ENV LD_PRELOAD=/root/miniconda/envs/torch_npu/lib/python3.9/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0