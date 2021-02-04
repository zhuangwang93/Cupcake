# train.py
#!/usr/bin/env	python3


import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import horovod.torch as hvd

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
sys.path.append("../../")
from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer


def memory_partition():
    optimizer.zero_grad()
    outputs = model(partition_inputs)
    loss = loss_function(outputs, partition_targets)
    loss.backward()
    torch.cuda.synchronize()
    start_time = time_()
    optimizer.step()
    torch.cuda.synchronize()

    return time_() - start_time


def train(epoch, tb=False):
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        step_start = time.time()
        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        step_finish = time.time()
        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(model.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=rank_train_size
        ))

        #update training loss for each iteration
        if tb:
            writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    if tb:
        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s, speed: {:.1f} img/sec'.format(epoch, finish - start, rank_train_size/(finish-start)))


@torch.no_grad()
def eval_training(epoch=0, tb=False):
    start = time.time()
    model.eval()

    test_loss = 0.0 # cost function error
    correct_1, correct_5 = 0.0, 0.0

    for (images, labels) in cifar100_test_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()

        _, pred = outputs.topk(5, 1, largest=True, sorted=True)
        labels = labels.view(labels.size(0), -1).expand_as(pred)
        correct = pred.eq(labels).float()

        #compute top 5
        correct_5 += correct[:, :5].sum()
        #compute top1
        correct_1 += correct[:, :1].sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Top-1 Accuracy: {:.4f}, Top-5 Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / rank_test_size,
        correct_1.float() / rank_test_size,
        correct_5.float() / rank_test_size,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct_1.float() / rank_test_size


if __name__ == '__main__':
    hvd.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='model to benchmark')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

    parser = add_parser_arguments(parser)
    args = parser.parse_args()

    model = get_network(args)
    torch.cuda.set_device(hvd.local_rank())
    model.cuda()
    cudnn.benchmark = True
    lr_scaler = hvd.size() if not args.use_adasum else 1

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    # data for memory partition
    partition_inputs, partition_targets = None, None
    for inputs, targets in cifar100_training_loader:
        partition_inputs, partition_targets = inputs.cuda(), targets.cuda()
        break
    rank_train_size = len(cifar100_training_loader.dataset) // hvd.size()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    rank_test_size = len(cifar100_test_loader.dataset) // hvd.size()

    loss_function = nn.CrossEntropyLoss()

    # Horovod: wrap optimizer with DistributedOptimizer.
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr * hvd.size())
    else:
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr * lr_scaler, weight_decay=5e-4)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader) / hvd.size()
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.model), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.model, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.model, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.model, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    input_tensor = input_tensor.cuda()
    writer.add_graph(model, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{model}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.model, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.model, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            model.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.model, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.model, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        model.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.model, recent_folder))

    if args.scheduler:
        if args.scheduler_baseline:
            grc.memory.clean()
            grc.compressor.clean()
            grc.memory.partition()
        else:
            schedule = Scheduler(grc, memory_partition, args)

    for epoch in range(1, args.epochs + 1):
        if epoch > args.warm:
            train_scheduler.step()

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.model, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.model, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)

    writer.close()
