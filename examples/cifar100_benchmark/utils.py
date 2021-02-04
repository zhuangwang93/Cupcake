""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import horovod.torch as hvd

def get_network(args):
    """ return given network
    """

    if args.model == 'vgg16':
        from models.vgg import vgg16_bn
        model = vgg16_bn()
    elif args.model == 'vgg13':
        from models.vgg import vgg13_bn
        model = vgg13_bn()
    elif args.model == 'vgg11':
        from models.vgg import vgg11_bn
        model = vgg11_bn()
    elif args.model == 'vgg19':
        from models.vgg import vgg19_bn
        model = vgg19_bn()
    elif args.model == 'densenet121':
        from models.densenet import densenet121
        model = densenet121()
    elif args.model == 'densenet161':
        from models.densenet import densenet161
        model = densenet161()
    elif args.model == 'densenet169':
        from models.densenet import densenet169
        model = densenet169()
    elif args.model == 'densenet201':
        from models.densenet import densenet201
        model = densenet201()
    elif args.model == 'googlenet':
        from models.googlenet import googlenet
        model = googlenet()
    elif args.model == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        model = inceptionv3()
    elif args.model == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        model = inceptionv4()
    elif args.model == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        model = inception_resnet_v2()
    elif args.model == 'xception':
        from models.xception import xception
        model = xception()
    elif args.model == 'resnet18':
        from models.resnet import resnet18
        model = resnet18()
    elif args.model == 'resnet34':
        from models.resnet import resnet34
        model = resnet34()
    elif args.model == 'resnet50':
        from models.resnet import resnet50
        model = resnet50()
    elif args.model == 'resnet101':
        from models.resnet import resnet101
        model = resnet101()
    elif args.model == 'resnet152':
        from models.resnet import resnet152
        model = resnet152()
    elif args.model == 'preactresnet18':
        from models.preactresnet import preactresnet18
        model = preactresnet18()
    elif args.model == 'preactresnet34':
        from models.preactresnet import preactresnet34
        model = preactresnet34()
    elif args.model == 'preactresnet50':
        from models.preactresnet import preactresnet50
        model = preactresnet50()
    elif args.model == 'preactresnet101':
        from models.preactresnet import preactresnet101
        model = preactresnet101()
    elif args.model == 'preactresnet152':
        from models.preactresnet import preactresnet152
        model = preactresnet152()
    elif args.model == 'resnext50':
        from models.resnext import resnext50
        model = resnext50()
    elif args.model == 'resnext101':
        from models.resnext import resnext101
        model = resnext101()
    elif args.model == 'resnext152':
        from models.resnext import resnext152
        model = resnext152()
    elif args.model == 'shufflenet':
        from models.shufflenet import shufflenet
        model = shufflenet()
    elif args.model == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        model = shufflenetv2()
    elif args.model == 'squeezenet':
        from models.squeezenet import squeezenet
        model = squeezenet()
    elif args.model == 'mobilenet':
        from models.mobilenet import mobilenet
        model = mobilenet()
    elif args.model == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        model = mobilenetv2()
    elif args.model == 'nasnet':
        from models.nasnet import nasnet
        model = nasnet()
    elif args.model == 'attention56':
        from models.attention import attention56
        model = attention56()
    elif args.model == 'attention92':
        from models.attention import attention92
        model = attention92()
    elif args.model == 'seresnet18':
        from models.senet import seresnet18
        model = seresnet18()
    elif args.model == 'seresnet34':
        from models.senet import seresnet34
        model = seresnet34()
    elif args.model == 'seresnet50':
        from models.senet import seresnet50
        model = seresnet50()
    elif args.model == 'seresnet101':
        from models.senet import seresnet101
        model = seresnet101()
    elif args.model == 'seresnet152':
        from models.senet import seresnet152
        model = seresnet152()
    elif args.model == 'wideresnet':
        from models.wideresidual import wideresnet
        model = wideresnet()
    elif args.model == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        model = stochastic_depth_resnet18()
    elif args.model == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        model = stochastic_depth_resnet34()
    elif args.model == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        model = stochastic_depth_resnet50()
    elif args.model == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        model = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return model


def get_training_dataloader(mean, std, batch_size=64, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(cifar100_training, num_replicas=hvd.size(), rank=hvd.rank())
    cifar100_training_loader = DataLoader(
        cifar100_training, num_workers=num_workers, batch_size=batch_size, sampler=train_sampler)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=256, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        cifar100_test, num_replicas=hvd.size(), rank=hvd.rank())
    cifar100_test_loader = DataLoader(
        cifar100_test, num_workers=num_workers, batch_size=batch_size, sampler=test_sampler)

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
