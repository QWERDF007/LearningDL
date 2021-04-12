# coding: utf-8

# This code extends the functionality of
# https://github.com/pytorch/examples/tree/master/imagenet
# to support cross-validation training, allowing you compute the out of sample
# predicted probabilities for the entire CIFAR training set:
# a necessary step for confident learning and the cleanlab package.
#
# Example showing how to obtain 4-fold cross-validated predicted probabilities:
# $ python3 imagenet_train_crossval.py \
#     -a resnet50 -b 256 --lr 0.1 --gpu 0 --cvn 4 --cv 0  \
#     --train-labels LABELS_PATH.json IMAGENET_PATH
# $ python3 imagenet_train_crossval.py \
#     -a resnet50 -b 256 --lr 0.1 --gpu 1 --cvn 4 --cv 1  \
# #   --train-labels LABELS_PATH.json IMAGENET_PATH
# $ python3 imagenet_train_crossval.py \
#     -a resnet50 -b 256 --lr 0.1 --gpu 2 --cvn 4 --cv 2  \
# #   --train-labels LABELS_PATH.json IMAGENET_PATH
# $ python3 imagenet_train_crossval.py \
#     -a resnet50 -b 256 --lr 0.1 --gpu 3 --cvn 4 --cv 3  \
# #   --train-labels LABELS_PATH.json IMAGENET_PATH
#
# Combine the cross-validation folds into a single predicted prob matrix
# $ python3 imagenet_train_crossval.py \
#     -a resnet50 --cvn 4 --combine-folds IMAGENET_PATH
#
# This script can also be used to train on CLEANED datasets, like this:
# python3 imagenet_train_crossval.py \
#     -a resnet50 -b 256 --lr 0.1 --gpu 0 --train-labels LABELS_PATH.json \
#     --dir-train-mask PATH_TO_CLEAN_DATA_BOOL_MASK.npy IMAGENET_PATH


# These imports enhance Python2/3 compatibility.
from __future__ import (
    print_function, absolute_import, division, unicode_literals, with_statement,
)
from efficientnet_pytorch import EfficientNet

import argparse
import copy
import json
import os
import random
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data.dataset import Dataset
import efficientnet_pytorch as efn

warnings.filterwarnings('ignore')

model_names = efn.VALID_MODELS
model_names += [name for name in models.__dict__
                if name.islower() and not name.startswith("__")
                and callable(models.__dict__[name])]
model_names = sorted(model_names)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-multi', '--multi', action='store_true', default=False,
                    help='whether multiple label or not')
parser.add_argument('-csv', '--csv', type=str, default=None,
                    help='path to annotation file')
parser.add_argument('-vcsv', '--val_csv', type=str, default=None,
                    help='path to val annotation file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='efficientnet-b0',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names))
parser.add_argument('-o', '--output', default='models', type=str, metavar='PATH',
                    help='output folder')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers ')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel'
                         'Use 128 when using Co-Teaching with --coteaching.')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay ',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency ')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint ')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=123, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--cv-seed', default=666, type=int,
                    help='seed for determining the cv folds. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cv', '--cv-fold', type=int, default=None,
                    metavar='N', help='The fold to holdout')
parser.add_argument('--cvn', '--cv-n-folds', default=0, type=int,
                    metavar='N', help='The number of folds')
parser.add_argument('-m', '--dir-train-mask', default=None, type=str,
                    help='Boolean mask with True for indices to '
                         'train with and false for indices to skip.')
parser.add_argument('--combine-folds', action='store_true', default=False,
                    help='Pass this flag and -a arch to combine probs from all'
                         'folds. You must pass -a and -cvn flags as well!')
parser.add_argument('--train-labels', type=str, default=None,
                    help='DIR of training labels format: json filename2integer')
parser.add_argument('--epoch-decay-start', type=int, default=80,
                    help='Co-Teaching number of epochs to train before'
                         'starting to decay the learning rate.')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Set to 50 if training on the first 50 classes.')
parser.add_argument('--turn-off-save-checkpoint', action='store_true',
                    help='Prevents saving model at every epoch of training.')
parser.add_argument('--webvision', action='store_true',
                    help='Normalize dataset based on webvision, not imagenet.')
parser.add_argument('--webvision-mini', action='store_true',
                    help='Normalize dataset based on webvision-mini.')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='multi labels threshold')

best_acc1 = 0


class MultiLabelDataset(Dataset):
    def __init__(self, root: str, df: pd.DataFrame, transform=None, target_transform=None):
        self.root = root
        paths = [os.path.join(self.root, name) for name in df.values[:, 0]]
        targets = np.array(df.values[:, 1:], dtype=np.float32)
        self.imgs = [(paths[i], targets[i]) for i in range(len(paths))]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        path, target = self.imgs[item]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def main(args):
    # 设置全部的随机种子，确保可复现
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # 设置gpu，取消分布式
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 获取当前环境下的gpu数量
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # 生成 ngpus_per_node 个进程，执行 main_worker，参数为args
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args), join=True)
    else:
        # Simply call main_worker function
        # 单进程
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    use_crossval = args.cvn > 0
    use_mask = args.dir_train_mask is not None
    cv_fold = args.cv  # k折中第几折
    cv_n_folds = args.cvn  # k
    class_weights = None

    if use_crossval and use_mask:
        raise ValueError(
            'Either args.cvn > 0 or dir-train-mask not None, but not both.')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # 分布式训练
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    # 创建模型，Efficientnet
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch.startswith('efficientnet'):
            model = EfficientNet.from_pretrained(
                args.arch, num_classes=args.num_classes)
        else:
            model = models.__dict__[args.arch](pretrained=True, num_classes=1000)
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                model.fc = nn.Linear(model.fc.in_features, args.num_classes)
            else:
                raise NotImplementedError("This arch doesn't implement pretrained")
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('efficientnet'):
            model = EfficientNet.from_name(args.arch, num_classes=args.num_classes)
        else:
            model = models.__dict__[args.arch](pretrained=True, num_classes=args.num_classes)

    if args.distributed:
        # For multiprocessing, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # 多进程训练，要在单块卡上构造DistributedDataParallel，否则单个进程会用完所有的gpu
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # 使用分布式训练时，每个进程只用一张卡时，要将batch-size划分到每块gpu
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size
            # to all available GPUs if device_ids are not set
            # 单进程多卡分布式时，会将batch-size均分给单进程的每块gpu
            model = torch.nn.parallel.DistributedDataParallel(model)
    # 单卡训练
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)  # 单独数据并行会比分布式数据并行曼
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        # momentum=args.momentum,
        # weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    # 恢复训练
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # In case you load checkpoint from different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.webvision:
        normalize = transforms.Normalize(mean=[0.5389, 0.5125, 0.4654],
                                         std=[0.2298, 0.2289, 0.2319])
    elif args.webvision_mini:
        normalize = transforms.Normalize(mean=[0.4849, 0.4773, 0.4130],
                                         std=[0.2089, 0.2036, 0.2046])
    else:  # imagenet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if args.multi:
        train_dataset = MultiLabelDataset(
            traindir,
            pd.read_csv(args.csv),
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    if os.path.exists(valdir):
        if args.multi:
            val_dataset = MultiLabelDataset(
                valdir,
                pd.read_csv(args.val_csv),
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]),
            )
        else:
            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
            )
    else:
        val_dataset = None

    verbose = (not args.multiprocessing_distributed) or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    # if training labels are provided use those instead of dataset labels
    # 用指定的labels文件替换ImageFolder生成的labels
    if args.train_labels is not None:
        with open(args.train_labels, 'r') as rf:
            train_labels_dict = json.load(rf)
        train_dataset.imgs = [(fn, train_labels_dict[fn]) for fn, _ in
                              train_dataset.imgs]
        train_dataset.samples = train_dataset.imgs

    # If training only on cross-validated portion & make val_set = train_holdout
    # 交叉验证，令val_set = train_holdout，即验证集为交叉验证的非训练那部分
    if use_crossval:
        checkpoint_fn = os.path.join(args.output, "model_{}__fold_{}__checkpoint.pth.tar".format(
            args.arch, cv_fold))
        if verbose:
            print('Computing fold indices. This takes 15 seconds.')
        # Prepare labels
        # 获取训练集的labels
        if args.multi:
            labels = [label for img, label in MultiLabelDataset(
                traindir, pd.read_csv(args.csv)).imgs]
        else:
            labels = [label for img,
                                label in datasets.ImageFolder(traindir).imgs]
        # Split train into train and holdout for particular cv_fold.
        if args.multi:
            kf = KFold(n_splits=cv_n_folds, shuffle=True,
                       random_state=args.cv_seed)
        else:
            kf = StratifiedKFold(n_splits=cv_n_folds,
                                 shuffle=True, random_state=args.cv_seed)
        cv_train_idx, cv_holdout_idx = (
            list(kf.split(range(len(labels)), labels))[cv_fold])
        # Separate datasets
        np.random.seed(args.cv_seed)
        holdout_dataset = copy.deepcopy(train_dataset)
        holdout_dataset.imgs = [train_dataset.imgs[i] for i in cv_holdout_idx]
        holdout_dataset.samples = holdout_dataset.imgs
        train_dataset.imgs = [train_dataset.imgs[i] for i in cv_train_idx]
        train_dataset.samples = train_dataset.imgs
        if verbose:
            print('Train size:', len(cv_train_idx), len(train_dataset.imgs))
            print('Holdout size:', len(cv_holdout_idx),
                  len(holdout_dataset.imgs))

        if val_dataset is None:
            # val_set = train_holdout
            val_dataset = copy.deepcopy(holdout_dataset)
            if verbose:
                print("Using training holdout set as validation set")
    else:
        checkpoint_fn = os.path.join(
            args.output, "model_{}__checkpoint.pth.tar".format(args.arch))
        if use_mask:
            checkpoint_fn = os.path.join(args.output, "model_{}__masked__checkpoint.pth.tar".format(
                args.arch))
            orig_class_counts = np.bincount(
                [lab for img, lab in datasets.ImageFolder(traindir).imgs])
            train_bool_mask = np.load(args.dir_train_mask)
            # Mask labels
            train_dataset.imgs = [img for i, img in
                                  enumerate(train_dataset.imgs) if
                                  train_bool_mask[i]]
            train_dataset.samples = train_dataset.imgs
            clean_class_counts = np.bincount(
                [lab for img, lab in train_dataset.imgs])
            print('Train size:', len(train_dataset.imgs))
            # Compute class weights to re-weight loss during training
            # Should use the confident joint to estimate the noise matrix then
            # class_weights = 1 / p(s=k, y=k) for each class k.
            # Here we approximate this with a simpler approach
            # class_weights = count(y=k) / count(s=k, y=k)
            class_weights = torch.Tensor(
                orig_class_counts / clean_class_counts)
            class_weights = class_weights.cuda(args.gpu)

    assert val_dataset is not None, "Check that you have 'val' dir or use cross validation"

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.distributed,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,  # Don't train on last epoch: could be 1 noisy example.
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    if args.multi:
        # define loss function (criterion)
        criterion = nn.BCELoss(weight=class_weights).cuda(args.gpu)
        # define separate loss function for val set that does not use class_weights
        val_criterion = nn.BCELoss(weight=None).cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)
        val_criterion = nn.CrossEntropyLoss(weight=None).cuda(args.gpu)

    if args.evaluate:
        validate(val_loader, model, criterion, verbose, args)
        return

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, verbose=verbose)

    for epoch in range(args.start_epoch, args.epochs):
        # 配合分布式sampler的shuffle，使每个训练进程能在每轮获取
        # 不同部分的样本，否则都是同一批数据
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        # adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, verbose, args)

        # Evaluate on validation set
        acc1, loss = validate(val_loader, model, val_criterion, verbose, args)

        # lr scheduler
        scheduler.step(loss)

        # Remember best acc@1, model, and save checkpoint.
        is_best = acc1 > best_acc1
        best_acc1 = max(best_acc1, acc1)

        if (
                not args.turn_off_save_checkpoint
                and not args.multiprocessing_distributed
        ) or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=is_best,
                output=args.output,
                filename=checkpoint_fn,
                cv_fold=cv_fold,
                use_mask=use_mask,
            )

    # 获取样本外的预测概率 out-of-sample predicted probabilities P_hat(k,i)
    # 交叉验证的保留部分的预测概率
    if use_crossval and verbose:
        holdout_loader = torch.utils.data.DataLoader(
            holdout_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        if not args.turn_off_save_checkpoint:  # Load best of saved checkpoints
            print("=> loading best model_{}__fold_{}_best.pth.tar".format(
                args.arch, cv_fold))
            checkpoint = torch.load(
                os.path.join(args.output, "model_{}__fold_{}_best.pth.tar".format(args.arch, cv_fold)))
            model.load_state_dict(checkpoint['state_dict'])
        print("Running forward pass on holdout set of size:",
              len(holdout_dataset.imgs))
        # predict，获取各类的概率
        probs = get_probs(holdout_loader, model, args)
        np.save(os.path.join(args.output, 'model_{}__fold_{}__probs.npy'.format(
            args.arch, cv_fold)), probs)

    print("=> GPU: {} barrier".format(args.gpu))
    #  pytorch 1.1 barrier 有 bug，rank 不为 0 的会直接结束，
    # 导致 0 的其余动作也会因为多进程的调度而无法进行
    dist.barrier()
    print("=> GPU: {} done".format(args.gpu))


def train(train_loader, model, criterion, optimizer, epoch, verbose, args):
    """
    训练
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    recall = AverageMeter()

    aver_acc = AverageMeter()
    aver_loss = AverageMeter()
    aver_recall = AverageMeter()

    world_size = torch.distributed.get_world_size()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # print(len(input), len(target), flush=True)
        # print(type(input))
        # print(input.shape)
        # if batch is size 1, skip because batch-norm will fail
        if len(input) <= 1:
            continue

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        if args.multi:
            output = torch.sigmoid(output)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if args.multi:
            precision, recall_score = calculate_metrics(output.detach().cpu().numpy(),
                                                        target.detach().cpu().numpy(), 0.5)
        else:
            precision = accuracy(output, target, topk=(1,))[0].item()
            recall_score = precision
        losses.update(loss.item(), input.size(0))
        top1.update(precision, input.size(0))
        recall.update(recall_score, input.size(0))
        # top5.update(acc5[0], input.size(0))

        running_metrics = torch.FloatTensor(
            [loss, precision, recall_score]).cuda(args.gpu)
        if args.rank >= 0:
            dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
            running_metrics = running_metrics / world_size
        aver_loss.update(running_metrics[0].item())
        aver_acc.update(running_metrics[1].item())
        aver_recall.update(running_metrics[2].item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % args.print_freq == 0:
            # print('[{0}] Epoch: [{1}][{2}/{3}]  '
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
            #       'Loss {loss.val:.4f} ({loss.avg:.4f}) ({aver_loss.avg:.4f})  '
            #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) ({aver_acc.avg:.3f})'.format(
            #     time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i, len(train_loader), batch_time=batch_time,
            #     data_time=data_time, loss=losses, top1=top1, aver_loss=aver_loss, aver_acc=aver_acc),
            #     flush=True)

            print('[{0}] Epoch: [{1}][{2}/{3}]  '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) ({aver_loss.avg:.4f})  '
                  'Percision {top1.val:.3f} ({top1.avg:.3f}) ({aver_acc.avg:.3f})  '
                  'Recall {recall.val:.3f} ({recall.avg:.3f}) ({aver_recall.avg:.3f})  '.format(
                time.strftime('%Y-%m-%d %H:%M:%S'), epoch, i, len(train_loader), batch_time=batch_time, loss=losses,
                top1=top1, recall=recall, aver_loss=aver_loss, aver_acc=aver_acc, aver_recall=aver_recall),
                flush=True)


def get_probs(loader, model, args):
    """
    预测loader中的样本，并将输出经过softmax，转换为numpy array
    """
    # Switch to evaluate mode.
    model.eval()
    n_total = len(loader.dataset.imgs) / float(loader.batch_size)
    outputs = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(loader):
            print("\rComplete: {:.1%}".format(i / n_total), end="")
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # compute output
            outputs.append(model(input))

    # Prepare outputs as a single matrix
    if args.multi:
        probs = np.concatenate([
            torch.nn.functional.sigmoid(z) if args.gpu is None else
            torch.nn.functional.sigmoid(z).cpu().numpy()
            for z in outputs
        ])
    else:
        probs = np.concatenate([
            torch.nn.functional.softmax(z, dim=1) if args.gpu is None else
            torch.nn.functional.softmax(z, dim=1).cpu().numpy()
            for z in outputs
        ])

    return probs


def validate(val_loader, model, criterion, verbose, args):
    """
    验证
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    recall = AverageMeter()

    aver_acc = AverageMeter()
    aver_loss = AverageMeter()
    aver_recall = AverageMeter()

    # switch to evaluate mode
    model.eval()

    world_size = dist.get_world_size()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            if args.multi:
                output = torch.sigmoid(output)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if args.multi:
                precision, recall_score = calculate_metrics(
                    output.cpu().numpy(), target.cpu().numpy(), 0.5)
            else:
                precision = accuracy(output, target, topk=(1,))[0].item()
                recall_score = precision
            losses.update(loss.item(), input.size(0))
            top1.update(precision, input.size(0))
            recall.update(recall_score, input.size(0))
            # top5.update(acc5[0], input.size(0))

            running_metrics = torch.FloatTensor(
                [loss, precision, recall_score]).cuda(args.gpu)
            if args.rank >= 0:
                dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
                running_metrics = running_metrics / world_size
            aver_loss.update(running_metrics[0].item())
            aver_acc.update(running_metrics[1].item())
            aver_recall.update(running_metrics[2].item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if verbose and i % args.print_freq == 0:
                print('[{0}] Test: [{1}/{2}]  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}) ({aver_loss.avg:.4f})  '
                      'Percision {top1.val:.3f} ({top1.avg:.3f}) ({aver_acc.avg:.3f})  '
                      'Recall {recall.val:.3f} ({recall.avg:.3f}) ({aver_recall.avg:.3f})  '.format(
                    time.strftime('%Y-%m-%d %H:%M:%S'), i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, recall=recall, aver_loss=aver_loss, aver_acc=aver_acc, aver_recall=aver_recall),
                    flush=True)

        if verbose:
            print(' * Acc@1 {aver_acc.avg:.3f} Loss {aver_loss.avg:.4f} Recall: {aver_recall.avg:.3f}'.format(
                aver_acc=aver_acc, aver_loss=aver_loss, aver_recall=aver_recall), flush=True)

    return aver_acc.avg, aver_loss.avg


def save_checkpoint(state, is_best, output, filename='checkpoint.pth.tar', cv_fold=None,
                    use_mask=False):
    """
    保存最优的模型
    """
    torch.save(state, filename)
    if is_best:
        sm = "__masked" if use_mask else ""
        sf = "__fold_{}".format(cv_fold) if cv_fold is not None else ""
        wfn = os.path.join(
            output, 'model_{}{}{}_best.pth.tar'.format(state['arch'], sm, sf))
        shutil.copyfile(filename, wfn)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    记录当前分值和计算评价值
    """

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    每30轮学习率降低为当前的0.1
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=np.float32)
    return (metrics.precision_score(y_true=target, y_pred=pred, average='weighted'),
            metrics.recall_score(y_true=target, y_pred=pred, average='weighted'))


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    计算指定k的准确率
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))  # 这里的准确率已经换算为百分比
        return res


def combine_folds(args):
    """
    合并k折交叉验证训练后得到的概率矩阵，得到整个训练集的概率矩阵
    """
    wfn = os.path.join(
        args.output, 'train__model_{}__pyx.npy'.format(args.arch))
    print('Make sure you specified the model architecture with flag -a.')
    print('This method will overwrite file: {}'.format(wfn))
    print('Computing fold indices. This takes 15 seconds.')
    # 获取训练集的标签
    # Prepare labels
    if args.multi:
        labels = [label for img, label in MultiLabelDataset(
            os.path.join(args.data, "train/"), pd.read_csv(args.csv)).imgs]
    else:
        labels = [label for img, label in datasets.ImageFolder(
            os.path.join(args.data, "train/")).imgs]
    # 初始化pyx矩阵 [Nxm]
    # Initialize pyx array (output of trained network)
    pyx = np.empty((len(labels), args.num_classes))

    # 对每个k将训练集分为train和test两部分，验证部分不重合，类别均衡，均为1/k，random_state保证结果一致
    # Split train into train and holdout for each cv_fold.
    if args.multi:
        kf = KFold(n_splits=args.cvn, shuffle=True, random_state=args.cv_seed)
    else:
        kf = StratifiedKFold(n_splits=args.cvn, shuffle=True,
                             random_state=args.cv_seed)

    # 读取每个k对应的验证集的概率合并，生成概率矩阵
    for k, (cv_train_idx, cv_holdout_idx) in enumerate(kf.split(range(len(labels)), labels)):
        probs = np.load(os.path.join(
            args.output, 'model_{}__fold_{}__probs.npy'.format(args.arch, k)))
        pyx[cv_holdout_idx] = probs[:, :args.num_classes]
    print('Writing final predicted probabilities.')
    np.save(wfn, pyx)

    # 计算准确率，计算即每个样本在概率矩阵中最大的值所对于的类作为预测标签与原始的标签相等的，再除以样本总数
    # Compute overall accuracy
    print('Computing Accuracy.', flush=True)
    if args.multi:
        acc, recall = calculate_metrics(pyx, labels, 0.5)
    else:
        acc = sum(np.array(labels) == np.argmax(
            pyx, axis=1)) / float(len(labels))
        recall = acc
    print('Accuracy: {:.4f} Recall: {:.4f}'.format(acc, recall))


if __name__ == '__main__':
    arg_parser = parser.parse_args()
    if arg_parser.webvision:
        # imagenet
        arg_parser.num_classes = 1000
    elif arg_parser.webvision_mini:
        # mini-imagenet
        arg_parser.num_classes = 50
    else:
        # 自定义分类数
        pass
    # 合并k折的结果
    if arg_parser.combine_folds:
        combine_folds(arg_parser)
    else:
        main(arg_parser)
