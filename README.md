<p align="center"> 
<img src="https://r.huaweistatic.com/s/ascendstatic/lst/header/header-logo.png" width="256px">
</p>

##### 基础环境

- 固件与驱动版本：[1.0.RC3.alpha](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=7.0.0.alpha002&driver=1.0.RC3.alpha)
- CANN 版本：[7.0.0.alpha002](https://www.hiascend.com/zh/developer/download/community/result?module=pt+tf+cann)
- Python 版本：[3.9.18](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
- Torch 版本：[2.1.0](https://gitee.com/link?target=https%3A%2F%2Fdownload.pytorch.org%2Fwhl%2Fcpu%2Ftorch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl%23sha256%3Dde7d63c6ecece118684415a3dbd4805af4a4c1ee1490cccf7405d8c240a481b4)
- Torch_NPU 版本：[2.1.0rc1](https://gitee.com/ascend/pytorch)

##### 环境安装

- 安装依赖
  ```shell
  sed -i 's/ports.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
  apt update
  apt install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev git wget curl
  ```

  ```shell
  pip install pip --upgrade 
  pip install attrs cython numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py -i https://pypi.tuna.tsinghua.edu.cn/simple
  pip install wheel pyyaml typing_extensions expecttest -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 安装 CANN
  ```shell
  chmod +x Ascend-cann-toolkit_7.0.RC1_linux-aarch64.run
  ./Ascend-cann-toolkit_7.0.RC1_linux-aarch64.run --install
  ```

  ```shell
  chmod +x Ascend-cann-kernels-910b_7.0.RC1_linux.run
  ./Ascend-cann-kernels-910b_7.0.RC1_linux.run --install
  ```

- 安装 Torch
  ```shell
  chmod +x torch-2.1.0-cp39-cp39-manylinux2014_aarch64.whl
  pip install torch-2.1.0-cp39-cp39-manylinux2014_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 安装 Torch_NPU
  ```shell
  chmod +x torch_npu-2.1.0rc1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
  pip install torch_npu-2.1.0rc1-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```


##### 使用 Docker

  ```shell
  docker pull xxx/xxx:latest
  ```

  ```shell
  docker run --network host --name my_docker --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -itd  xxx/xxx:latest bash
  ```

##### Hugging Face 工具链及模型支持状态

|               |                                                                                                                                                                                                                                                                                                                                                 Transformers                                                                                                                                                                                                                                                                                                                                                 |        Accelerate        |        PEFT        |        TRL        |
| :-----------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: | :----------------: | :---------------: |
| Support State | [![Transformers NPU Build](https://github.com/zhangsibo1129/hf-ci-demo/actions/workflows/transformers-npu-build.yml/badge.svg)](https://github.com/zhangsibo1129/hf-ci-demo/actions/workflows/transformers-npu-build.yml) <br> [![Transformers NPU Inference](https://github.com/zhangsibo1129/hf-ci-demo/actions/workflows/transformers-npu-inference.yml/badge.svg)](https://github.com/zhangsibo1129/hf-ci-demo/actions/workflows/transformers-npu-inference.yml) <br> [![Transformers NPU Training](https://github.com/zhangsibo1129/hf-ci-demo/actions/workflows/transformers-npu-training.yml/badge.svg)](https://github.com/zhangsibo1129/hf-ci-demo/actions/workflows/transformers-npu-training.yml) |                          |                    |                   |
|  [Llama2]()   |                                                                                                                                                                                                                                                                                                                                          <!--transformers-Llama2-->                                                                                                                                                                                                                                                                                                                                          | <!--accelerate-Llama2--> | <!--peft-Llama2--> | <!--trl-Llama2--> |
|   [Bert]()    |                                                                                                                                                                                                                                                                                                                                          <!--transformers-Bert-->✅                                                                                                                                                                                                                                                                                                                                           |  <!--accelerate-Bert-->  |  <!--peft-Bert-->  |  <!--trl-Bert-->  |
|   [BLOOM]()   |                                                                                                                                                                                                                                                                                                                                          <!--transformers-BLOOM-->                                                                                                                                                                                                                                                                                                                                           | <!--accelerate-BLOOM-->  | <!--peft-BLOOM-->  | <!--trl-BLOOM-->  |
|   [GPT-2]()   |                                                                                                                                                                                                                                                                                                                                          <!--transformers-GPT-2-->                                                                                                                                                                                                                                                                                                                                           | <!--accelerate-GPT-2-->  | <!--peft-GPT-2-->  | <!--trl-GPT-2-->  |
|    [OPT]()    |                                                                                                                                                                                                                                                                                                                                           <!--transformers-OPT-->✅                                                                                                                                                                                                                                                                                                                                           |  <!--accelerate-OPT-->   |  <!--peft-OPT-->   |  <!--trl-OPT-->   |
|    Others     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                          |                    |                   |