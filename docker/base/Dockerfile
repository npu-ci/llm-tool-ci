FROM ubuntu:20.04

# Install Dependencies
RUN sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    gcc \
    g++ \
    make \
    cmake \
    zlib1g \
    zlib1g-dev \
    openssl \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    unzip \
    pciutils \
    net-tools \
    libblas-dev \
    gfortran \
    libblas3 \
    libopenblas-dev \
    git \
    wget \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python
ENV MINICONDA_FILE=Miniconda3-latest-Linux-aarch64.sh
ENV MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

RUN wget -O ${MINICONDA_FILE} ${MINICONDA_URL} && \
    chmod +x ${MINICONDA_FILE} && \
    bash ${MINICONDA_FILE} -b -p /root/miniconda && \
    /root/miniconda/bin/conda create --name torch_npu -y python=3.9 && \
    rm -f ${MINICONDA_FILE}

ENV PATH=/root/miniconda/envs/torch_npu/bin/:$PATH

# Install Python Packages
ENV PIP_SOURCE_URL=https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install pip --no-cache-dir --upgrade -i ${PIP_SOURCE_URL} && \
    pip install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py -i ${PIP_SOURCE_URL} && \
    pip install wheel pyyaml typing_extensions expecttest -i ${PIP_SOURCE_URL}

# Install CANN toolkit
ENV CANN_TOOLKIT_FILE=Ascend-cann-toolkit_7.0.0.alpha002_linux-aarch64.run
ENV CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C15SPC702/Ascend-cann-toolkit_7.0.0.alpha002_linux-aarch64.run?response-content-type=application/octet-stream
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

RUN wget -O ${CANN_TOOLKIT_FILE} ${CANN_TOOLKIT_URL} && \
    chmod +x ${CANN_TOOLKIT_FILE} && \
    sh -c  '/bin/echo -e "Y" | ./${CANN_TOOLKIT_FILE} --install' && \
    rm -f ${CANN_TOOLKIT_FILE}

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# Install CANNN Kernels
ENV CANN_KERNELS_FILE=Ascend-cann-kernels-910b_7.0.0.alpha002_linux.run
ENV CANN_KERNELS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C15SPC702/Ascend-cann-kernels-910b_7.0.0.alpha002_linux.run?response-content-type=application/octet-stream

RUN wget -O ${CANN_KERNELS_FILE} ${CANN_KERNELS_URL} && \
    chmod +x ${CANN_KERNELS_FILE} && \
    sh -c  '/bin/echo -e "Y" | ./${CANN_KERNELS_FILE} --install' && \
    rm -f ${CANN_KERNELS_FILE}

# Install Pytorch
ENV TORCH_FILE=torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
ENV TORCH_URL=https://download.pytorch.org/whl/cpu/torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

RUN wget -O ${TORCH_FILE} ${TORCH_URL} && \
    chmod +x ${TORCH_FILE} && \
    pip install ${TORCH_FILE} -i ${PIP_SOURCE_URL} && \
    rm -f ${TORCH_FILE}

# Install Torch NPU
RUN pip install torch-npu==2.1.0rc1 -i ${PIP_SOURCE_URL}

# avoid import torch_npu error
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# Install Rust
ENV RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
ENV RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup

RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH=/root/.cargo/bin:$PATH

RUN echo "[source.crates-io]\n"\
"replace-with = 'ustc'\n"\
"\n"\
"[source.ustc]\n"\
"registry = \"sparse+https://mirrors.ustc.edu.cn/crates.io-index/\"\n">/root/.cargo/config

# avoid import torch_npu error (from Install Rust)
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
