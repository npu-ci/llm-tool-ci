FROM zhangsibo1129/ubuntu-cann-torch21-py39:latest

ARG REF=main
WORKDIR /root

RUN git clone https://github.com/huggingface/peft.git && \
    cd peft && \
    git checkout $REF

ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN cd peft && \
    pip install -e ".[test]" -i ${PIP_SOURCE_URL} \
    pip install evaluate scikit-learn -i ${PIP_SOURCE_URL}

# avoid import error caused by scikit-learn
ENV PYTHON_VERSION=3.9
ENV CONDA_ENV_NAME=torch_npu
ENV LD_PRELOAD=/root/miniconda/envs/${CONDA_ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:${LD_PRELOAD}
