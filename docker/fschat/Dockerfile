FROM zhangsibo1129/ubuntu-cann-torch21-py39:latest

ARG REF=main
WORKDIR /root

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/lm-sys/FastChat.git && \
    cd FastChat && \
    git checkout $REF

ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install --no-cache-dir -e ".[model_worker,webui]" -i ${PIP_SOURCE_URL}
