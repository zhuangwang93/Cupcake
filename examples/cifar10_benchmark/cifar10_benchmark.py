'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import horovod.torch as hvd
import timeit
from time import time as time_

from models import *
import torchvision
import torchvision.transforms as transforms
from torchvision import models

import os
import sys
import argparse

from utils import progress_bar
sys.path.append("../../")
from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer
from mergeComp_dl.torch.scheduler.scheduler import Scheduler


# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser = add_parser_arguments(parser)
args = parser.parse_args()

hvd.init()
# Model
print('==> Building model..')
# model = VGG('VGG19')
#model = ResNet18()
# model = PreActResNet18()
# model = GoogLeNet()
# model = DenseNet121()
# model = ResNeXt29_2x64d()
# model = MobilNet()
# model = MobileNetV2()
# model = DPN92()
# model = ShuffleNetG2()
# model = SENet18()
# model = ShuffleNetV2(1)
# model = EfficientNetB0()
# model = RegNetX_200MF()
#model = SimpleDLA()
#model = models.resnet50()
if args.model == "resnet18":
    model = ResNet18()
elif args.model == 'resnet50':
    model = ResNet50()
elif args.model == 'resnet152':
    model = ResNet152()
else:
    sys.exit("Unknown model")

args.cuda = not args.no_cuda and torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

lr_scaler = hvd.size() if not args.use_adasum else 1
if args.cuda:
    torch.cuda.set_device(hvd.local_rank())
    model.cuda()
    #model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = hvd.local_size()

torch.set_num_threads(1)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data/cifar10-data-%d' % hvd.rank(), train=True, download=True, transform=transform_train)

# Horovod: use DistributedSampler to partition the training data.
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data/cifar10-data-%d' % hvd.rank(), train=False, download=True, transform=transform_test)
# Horovod: use DistributedSampler to partition the test data.
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')



if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
# Horovod: wrap optimizer with DistributedOptimizer.
if args.adam:
    optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())
else:
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr * lr_scaler, weight_decay=5e-4)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

optimizer, grc = wrap_compress_optimizer(model, optimizer, args)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# data for memory partition
partition_inputs, partition_targets = None, None
for inputs, targets in train_loader:
    partition_inputs, partition_targets = inputs.cuda(), targets.cuda()
    break

def memory_partition():
    optimizer.zero_grad()
    outputs = model(partition_inputs)
    loss = criterion(outputs, partition_targets)
    torch.cuda.synchronize()
    start_time = time_()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    return time_() - start_time


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time, end_time = 0, 0
    start_idx = 1
    end_idx = 149
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # the first few iterations are unstable
        if batch_idx == start_idx:
            torch.cuda.synchronize()
            start_time = time_()
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx == end_idx:
            torch.cuda.synchronize()
            end_time = time_()
        #if args.speed_test:
        #    continue
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if args.speed_test:
        print("Epoch: {}\tIterations: {}\tTime: {:.2f} s\tTraining speed: {:.1f} img/s".format(
            epoch,
            len(train_loader),
            end_time - start_time,
            (end_idx-start_idx+1)*args.batch_size/(end_time - start_time)
        ))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if hvd.rank() == 0:
                progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if args.scheduler:
    if args.scheduler_baseline:
        grc.memory.clean()
        grc.compressor.clean()
        grc.memory.partition()
    else:
        schedule = Scheduler(grc, memory_partition, args)

for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    if not args.speed_test:
        test(epoch)
        scheduler.step()
        if args.compress:
            grc.memory.update_lr(optimizer.param_groups[0]['lr'])
