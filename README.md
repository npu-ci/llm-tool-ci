<!-- <p align="center"> 
<img src="https://r.huaweistatic.com/s/ascendstatic/lst/header/header-logo.png" width="256px">
</p> -->

##### Hugging Face 工具链及模型支持状态

|  | Transformers | Accelerate | PEFT | FastChat |
| :---: | :---: | :---: | :---: | :---: |
| Support State | [![Transformers Check Environment](https://github.com/npu-ci/llm-tool-ci/actions/workflows/transformers-check.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/transformers-check.yml) <br> [![Transformers Inference](https://github.com/npu-ci/llm-tool-ci/actions/workflows/transformers-inference.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/transformers-inference.yml) <br> [![Transformers Training](https://github.com/npu-ci/llm-tool-ci/actions/workflows/transformers-training.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/transformers-training.yml) | [![Accelerate NPU Build](https://github.com/npu-ci/llm-tool-ci/actions/workflows/accelerate-build.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/accelerate-build.yml) <br> [![Accelerate NPU UnitTest](https://github.com/npu-ci/llm-tool-ci/actions/workflows/accelerate-unittest.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/accelerate-unittest.yml) |  [![PEFT Official Tests](https://github.com/npu-ci/llm-tool-ci/actions/workflows/peft-official-tests.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/peft-official-tests.yml) | [![FastChat NPU Deploy](https://github.com/npu-ci/llm-tool-ci/actions/workflows/fastchat-deploy.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/fastchat-deploy.yml) <br> [![FastChat NPU FineTune](https://github.com/npu-ci/llm-tool-ci/actions/workflows/fastchat-finetune.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/fastchat-finetune.yml) <br> [![FastChat NPU Inference](https://github.com/npu-ci/llm-tool-ci/actions/workflows/fastchat-inference.yml/badge.svg)](https://github.com/npu-ci/llm-tool-ci/actions/workflows/fastchat-inference.yml) |
| [Llama2]() |<!--transformers-Llama2--> |<!--accelerate-Llama2--> |<!--peft-Llama2--> |<!--fschat-Llama2--> |
| [Bert]() |<!--transformers-Bert--> |<!--accelerate-Bert--> |<!--peft-Bert--> |<!--fschat-Bert--> |
| [BLOOM]() |<!--transformers-BLOOM--> |<!--accelerate-BLOOM--> |<!--peft-BLOOM-->✅|<!--fschat-BLOOM--> |
| [GPT-2]() |<!--transformers-GPT-2--> |<!--accelerate-GPT-2--> |<!--peft-GPT-2-->✅|<!--fschat-GPT-2--> |
| [OPT]() |<!--transformers-OPT--> |<!--accelerate-OPT--> |<!--peft-OPT-->✅|<!--fschat-OPT--> |
| [GPT-NEO]() |<!--transformers-GPT-NEO--> |<!--accelerate-GPT-NEO--> |<!--peft-GPT-NEO-->|<!--fschat-GPT-NEO--> |
| [GPT-J]() |<!--transformers-GPT-J--> |<!--accelerate-GPT-J--> |<!--peft-GPT-J-->|<!--fschat-GPT-J--> |
| [FUYU]() |<!--transformers-FUYU--> |<!--accelerate-FUYU--> |<!--peft-FUYU-->|<!--fschat-FUYU--> |
| [GPT-NEOX]() |<!--transformers-GPT-NEOX--> |<!--accelerate-GPT-NEOX--> |<!--peft-GPT-NEOX-->|<!--fschat-GPT-NEOX--> |
| [Mistral]() |<!--transformers-Mistral--> |<!--accelerate-Mistral--> |<!--peft-Mistral-->|<!--fschat-Mistral--> |
| Others |  |  |  |  |

##### 环境依赖

- [固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=7.0.0.alpha002&driver=1.0.RC3.alpha)
- [CANN](https://www.hiascend.com/developer/download/community/result?module=cann)
- [Python](https://www.python.org/)
- [Torch](https://pytorch.org/)
- [Torch_NPU](https://gitee.com/ascend/pytorch/releases)

##### 最新环境安装(以Ubuntu,910b芯片演示)
- 最新依赖版本
  ```shell
  export NPU_DRIVER_VERSION=910b-npu-driver_23.0.rc3_linux-aarch64
  export NPU_FIRMWARE_VERSION=910b-npu-firmware_6.4.0.4.220
  export PYTHON_VERSION=3.9.18
  export CANN_TOOLKIT_VERSION=7.0.0_linux-aarch64
  export CANN_KERNELS_VERSION=910b_7.0.0_linux
  export TORCH_VERSION=2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64
  export TORCH_NPU_VERSION=2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64
  ```
- 安装包下载    
  [npu-driver](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2023.0.RC3/Ascend-hdk-910b-npu-driver_23.0.rc3_linux-aarch64.run)    
  [npu-firmware](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2023.0.RC3/Ascend-hdk-910b-npu-firmware_6.4.0.4.220.run)    
  [miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-aarch64.sh)    
  [CANN Toolkit](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run?response-content-type=application/octet-stream)    
  [CANN Kernerls](Ascend-cann-kernels-910b_7.0.0_linux.run)    
  [Torch](https://download.pytorch.org/whl/cpu/torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)
  [Torch_NPU](https://gitee.com/ascend/pytorch/releases/download/v5.0.0-pytorch2.1.0/torch_npu-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl)   
- 
- 安装固件与驱动

  ```shell
  # 创建驱动运行用户HwHiAiUser（运行驱动进程的用户），安装驱动时无需指定运行用户，默认即为HwHiAiUser
  groupadd HwHiAiUser
  useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
  # 安装
  chmod +x Ascend-hdk-${NPU_DRIVER_VERSION}.run
  chmod +x Ascend-hdk-${NPU_FIRMWARE_VERSION}.run
  ./Ascend-hdk-${NPU_DRIVER_VERSION}.run --full --install-for-all
  ./Ascend-hdk-${NPU_FIRMWARE_VERSION}.run --full
  ```
  ```shell
  # 重启
  reboot
  ```
  ```shell
  # 验证是否安装成功
  npu-smi info
  ```
- 安装依赖
    ```shell
    sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
    apt update
    apt install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev git wget curl
    ```

- 安装python
  ```shell
  ./Miniconda3-latest-Linux-aarch64.sh
  conda create --name torch_npu python=${PYTHON_VERSION}
  conda activate torch_npu
  echo conda activate torch_npu>>~/.bashrc
  
  pip install pip --upgrade 
  pip install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install wheel pyyaml typing_extensions expecttest -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

  
- 安装 CANN
  ```shell
  export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
  chmod +x Ascend-cann-toolkit_${CANN_TOOLKIT_VERSION}.run
  ./Ascend-cann-toolkit_${CANN_TOOLKIT_VERSION}.run --install
  source /usr/local/Ascend/ascend-toolkit/set_env.sh

  chmod +x Ascend-cann-kernels-${CANN_KERNELS_VERSION}_linux.run
  ./Ascend-cann-kernels-${VERSION}_linux.run --install
  ```

- 安装 Torch
      
  ```shell
  chmod +x torch-${TORCH_VERSION}.whl
  pip install torch-${TORCH_VERSION}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 安装 Torch_npu
  ```shell
  chmod +x torch_npu-${TORCH_NPU_VERSION}.whl
  pip install torch_npu-${TORCH_NPU_VERSION}.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  ```


##### 使用 Docker 镜像

- 从 Dockerfile 构建
  ```shell
  git clone https://github.com/npu-ci/llm-tool-ci.git
  cd llm-tool-ci
  docker build -t your_docker_image -f ./Dockerfile  .
  ```

- 从 Docker Hub 拉取

  ```shell
  docker pull zhangsibo1129/ubuntu-cann-torch21-py39:latest
  ```

- Docker 启动命令
  ```shell
  docker run -itd \
  --name your_docker_name \
  --network host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  your_docker_image bash
  ```

- 进入 Docker 环境
  ```shell
  docker exec -it your_docker_name bash
  ```