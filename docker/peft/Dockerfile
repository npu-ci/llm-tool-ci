FROM zhangsibo1129/ubuntu-cann-torch21-py39:latest

ARG REF=main
WORKDIR /root
ENV PYTHON_VERSION=3.9
ENV CONDA_ENV_NAME=torch_npu
ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN git clone https://github.com/npu-ci/llm-tool-ci.git

RUN git clone https://github.com/huggingface/peft.git && \
    cd peft && \
    git checkout $REF

RUN cd peft && \
    pip install -e .[test] -i ${PIP_SOURCE_URL} \
    pip install evaluate scikit-learn -i ${PIP_SOURCE_URL}
ENV export LD_PRELOAD=$LD_PRELOAD:/root/miniconda/envs/${CONDA_ENV_NAME}/lib/python${PYTHON_VERSION}/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0