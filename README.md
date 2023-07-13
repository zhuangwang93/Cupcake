# Cupcake: A Compression Scheduler for Scalable Communication-Efficient Distributed Training

## Overview

Cupcake is a compression scheduler to optimize the scalability of communication-efficient distributed training. It provides a general framework and implementations for several popular compression algorithms. Cupcake automatically schedules the compression operations to optimize the performance of compression algorithms without the knowledge of model architectures or system parameters. It can significantly reduce the compression overhead and meanwhile overlap the communication with the computation to reduce the communication overhead. We release the PyTorch version of the Cupcake.

## Citation

If you find our project useful in your research, please consider citing:

```
@article{wang2023cupcake,
  title={Cupcake: A Compression Scheduler for Scalable Communication-Efficient Distributed Training},
  author={Wang, Zhuang and Wu, Xinyu and Xu, Zhaozhuo and Ng, TS},
  journal={Proceedings of Machine Learning and Systems},
  volume={5},
  year={2023}
}
```


## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.7
- [PyTorch](https://github.com/pytorch/pytorch)
- [Horovod](https://github.com/horovod/horovod) 
- [numpy](https://github.com/numpy/numpy)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [openmpi](https://www.open-mpi.org/software/ompi/)
- [NCCL](https://github.com/NVIDIA/nccl)
- [CuPy](https://docs.cupy.dev/en/stable/install.html) - Depending on your CUDA Toolkit version

## Install

```shell script
# Step 1: install pytorch: https://pytorch.org/get-started/locally/
# Step 2: install horovod
HOROVOD_GPU_OPERATIONS=NCCL install --no-cache-dir horovod
horovodrun --check-build
# Step 3: download Cupcake
git clone https://github.com/zhuangwang93/Cupcake.git
```


## Main Components
The four main components of Cupcake framework are the `Communicator`, `Compressor`, `Memory` and `Scheduler` abstract classes.
- `Communicator` implementations define the communication primitives used by Cupcake.
- `Compressor` implementations provide different `compress` and `decompress` operations for various gradient compression algorithms.
- `Memory` implementations provide the `partition`, `update`, `compensate` and `reduce` methods for memory.
- `Scheduler` implementations provide the optimization scheduler for performance of training speed (obsolete).

Cupcake can support following configurations and we are extending it for more options.

| Configurations | Options                                                                            |
| -------------- | ---------------------------------------------------------------------------------- |
| Communicator   | Allreduce, Allgather                                                               |
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
Real examples are provided under the [examples](examples) folder. We provide two kinds of benchmarks for the evaluation. The models include ResNet50 and ResNet101 on CIFAR10 and ImageNet.

For CIFAR10/CIFAR100 benchmark, the model and dataset could be download automatically. There is no additional operation for CIFAR10/CIFAR100.

For ImageNet benchmark, you need to download the ImageNet dataset for the evaluation.
