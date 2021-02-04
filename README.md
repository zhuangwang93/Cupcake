# MergeComp - A Compression Scheduler for Scalable Communication-Efficient Distributed Training

## Overview

MergeComp is a compression scheduler to optimize the scalability of communication-efficient distributed training. It provides a general framework and implementations for several popular compression algorithms. MergeComp automatically schedules the compression operations to optimize the performance of compression algorithms without the knowledge of model architectures or system parameters. It can significantly reduce the compression overhead and meanwhile overlap the communication with the computation to reduce the communication overhead. We release the PyTorch version of the MergeComp.

## Content

- [Prerequisites](#prerequisites)
- [Install](#install)
- [Main Components](#main-components)
- [Training Examples](#training-examples)
- [Performance](#performance)

## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.7.1
- [Horovod](https://github.com/horovod/horovod) >= 0.21.1
- [numpy](https://github.com/numpy/numpy) >= 1.19.2
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 1.2
- [tqdm](https://github.com/tqdm/tqdm) >= 4.56.0
- [openmpi](https://www.open-mpi.org/software/ompi/) >= 4.0.2
- [NCCL](https://github.com/NVIDIA/nccl) >= 2.8.3
- [CuPy](https://docs.cupy.dev/en/stable/install.html) - Depending on your CUDA Toolkit version

## Install

We provide a step-by-step tutorial for you to install each requirements and run benchmark training tasks. The simplest way to start is to use docker images.

### Docker

Nvidia docker install:
```shell script
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker

# Check docker: 
sudo docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```
We use [Horovod](https://github.com/horovod/horovod) to run distributed deep learning.
To use MergeComp via Docker, just pull the latest Horovod image:
```shell script
docker pull horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

sudo docker run -it --privileged -v ~/mergeComp:/home --net host --gpus all imageID
```

### Download mergeComp
```shell script
git clone https://github.com/Crystal-wxy/mergeComp.git
```

### CMake and OpenMPI [Optional]
You don't bother to install them if you are using Horovod docker
 ```shell script
# CMake install
wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz
sudo ./bootstrap
sudo make
sudo sudo make install

# OpenMPI install
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz
./configure
sudo make all install
```

### Conda 
If you want to create a new conda environment for mergeComp, please follow the following steps:
 ```shell script
conda create -n mergeComp python=3.7

conda activate mergeComp

# PyTorch install
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch -y

# Horovod + MPI install
HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod

# NCCL install
cd ~
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build
sudo apt install build-essential devscripts debhelper fakeroot
make pkg.debian.build
ls build/pkg/deb/

# Horovod + NCCL install
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=~/nccl/build HOROVOD_WITHOUT_GLOO=1  pip install --no-cache-dir horovod

# Check Horovod
horovodrun --check-build
```

### Potential isssus & solutions
- **Nvidia driver broken**: Refer to https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html -> Package managers

- **No enough shared memory in docker containers**: Refer to https://www.deepanseeralan.com/tech/changing-shmem-size-of-docker-container/


## Main Components
The four main components of mergeComp framework are the `Communicator`, `Compressor`, `Memory` and `Scheduler` abstract classes.
- `Communicator` implementations define the communication primitives used by mergeComp.
- `Compressor` implementations provide different `compress` and `decompress` operations for various gradient compression algorithms.
- `Memory` implementations provide the `partition`, `update`, `compensate` and `reduce` methods for memory.
- `Scheduler` implementations provide the optimization scheduler for performance of training speed.

MergeComp can support following configurations and we are extending it for more options.

| Configurations | Options                                                                            |
| -------------- | ---------------------------------------------------------------------------------- |
| Communicator   | Allreduce, Allgather, PS                                                           |
| Compressors    | FP32, FP16, DGC, Top-k, Rand-k, QSGD, EFSignSGD, SignSGD, TernGrad, OneBit, SigNUM |
| Memory         | Residual, DGC, None                                                                |

## Training Examples

### Quick start

Basic benchmark is provided in `compress_benchmark.py`. 
- For example, we can use the following command to run the benchmark on 4 GPUs, with compression algorithm as efsignsgd, communication primitive as allgather, memory as residual.
 ```shell script
 horovodrun -np 4 python compress_benchmark.py --compress --compressor efsignsgd --comm allgather --memory residual --fusion-num 1 --model=resnet50
  ```
- fusion-num specifies the number of partition groups. --fusion-num 0 is the layer-wise compression.

### Standard benchmarks
Real examples are provided under the [examples](examples) folder. We provide three kind of benchmarks for the evaluation. The models include ResNet50 and ResNet101 on CIFAR10 and ImageNet; Mask R-CNN on COCO.

For each benchmark, we provide running scripts to run examples over both PCIe and NVLink:
```shell script
bash run_pcie.sh
bash run_nvlink.sh
```

For CIFAR10/CIFAR100 benchmark, the model and dataset could be download automatically. There is no additional operation for CIFAR10/CIFAR100.

For ImageNet benchmark, you need to download the ImageNet dataset for the evaluation.

For COCO benchmark, you need to install detectron2 and download COCO dataset:
```shell script
# Install detectron2
cd -
python -m pip install -e detectron2

# download Coco dataset
bash download_dataset.sh

# Set dataset
export DETECTRON2_DATASETS=/data

# Run the scripts
./mask_rcnn.py --num-gpus 4 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml
```


## Performance

We show our experiments on distribute deep training for various compression algorithm
over both PCIe and NVLink. 

All experiments are conducted on a server equipped with 8 GPUs (NVIDIA Tesla V100 with 32 GB memory), two 20-core/40-thread processors (Intel Xeon Gold 6230 2.1GHz), PCIe 3.0 x 16, and NVLink. The server has an Ubuntu 18.04.4 LTS system. The default batch size for image classification is 64 and image segmentation is 1.

We compare mergeComp against FP32 (baseline) and layer-wise compression. Our evaluations show that MergeComp can improve the scalability of distributed training by more than 200% without losing accuracy for representative DNN models. It can even achieve scalability up to 99% over high-speed networks.

![resnet50_speed](https://github.com/zhuangwang93/fast_compression/blob/mergeComp/data/resnet50_results.png)
![resnet101_speed](https://github.com/zhuangwang93/fast_compression/blob/mergeComp/data/resnet101_results.png)

