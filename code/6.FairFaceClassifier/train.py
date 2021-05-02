import os
import copy
import argparse
import random
import logging
import warnings

import yaml
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from model import MODEL_NAMES, create_model, save_checkpoint
from helper import setup_default_logging, AverageMeter, calculate_metrics, accuracy
from dataset import create_dataset

warnings.filterwarnings('ignore')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--vdata', type=str, default=None,
                    help='path to validation data')
parser.add_argument('--multi', action='store_true', default=False,
                    help='train multiple labels')
parser.add_argument('-csv', '--csv', type=str, default=None, metavar='PATH',
                    help='path to annotation file')
parser.add_argument('--val-csv', type=str, default=None, metavar='PATH',
                    help='path to val annotation file')
parser.add_argument('--test-split', type=float, default=0.4,
                    help='ratio of validation')
parser.add_argument('--arch', default='resnet50', type=str, metavar='MODEL', choices=MODEL_NAMES,
                    help='model architecture: ' + ' | '.join(MODEL_NAMES))
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='multi labels threshold')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--patience-epochs', type=int, default=5, metavar='N',
                    help='patience epochs for Plateau LR scheduler')
parser.add_argument('--initial-checkpoint', default=None, type=str, metavar='PATH',
                    help='Initialize model from this checkpoint')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--distributed', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:12345', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()

    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, 'args.yaml'), 'w') as f:
        f.write(args_text)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # 分布式训练，单机多卡，多机多卡
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, join=True, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    setup_default_logging()
    _logger = logging.getLogger('train')

    if args.gpu is not None:
        _logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    args.verbose = not args.distributed or (args.distributed and args.rank % ngpus_per_node == 0)

    if args.verbose:
        _logger.info("create model {}".format(args.arch))
    model = create_model(args.arch, args.num_classes, args.pretrained)

    if args.distributed:
        # 单进程单卡训练
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # 单进程多卡训练
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.verbose:
            _logger.warning("no gpu for training, using cpu")

    optimizer = optim.Adam(model.parameters(), args.lr)

    start_epoch = args.start_epoch

    if args.initial_checkpoint is not None:
        if os.path.isfile(args.initial_checkpoint):
            if args.verbose:
                _logger.info("initializing model from '{}'".format(args.initial_checkpoint))
            if args.gpu is None:
                checkpoint = torch.load(args.initial_checkpoint)
            else:
                checkpoint = torch.load(args.initial_checkpoint, map_location='cuda:{}'.format(args.gpu))
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            if args.verbose:
                _logger.info("initialized model from '{}'".format(args.initial_checkpoint))

    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.verbose:
                _logger.info("loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.verbose:
                _logger.info("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.patience_epochs, verbose=args.verbose)

    if args.vdata is not None and args.val_csv is not None:
        train_df = pd.read_csv(args.csv)
        train_set = create_dataset(args.data, train_df, args.mean, args.std, args.multi, evaluate=False)
        val_df = pd.read_csv(args.val_csv)
        val_set = create_dataset(args.vdata, val_df, args.mean, args.std, args.multi, evaluate=False)
    else:
        if args.multi:
            assert args.csv is not None, "Please specify annotation file"
            df = pd.read_csv(args.csv)
            val_df = df.sample(frac=args.test_split, random_state=args.seed)
            train_df = df.drop(val_df.index)
            train_set = create_dataset(args.data, train_df, args.mean, args.std, multi=True, train=True, evaluate=False)
            val_set = create_dataset(args.data, val_df, args.mean, args.std, multi=True, train=True, evaluate=False)
        else:
            df = pd.read_csv(args.csv) if args.csv is not None else None
            dataset = create_dataset(args.data, df, args.mean, args.std, multi=False, train=True, evaluate=False)
            train_set = copy.deepcopy(dataset)
            val_set = copy.deepcopy(dataset)
            kf = StratifiedShuffleSplit(n_splits=1, test_size=args.test_split, random_state=args.seed)
            train_idx, val_idx = next(kf.split(dataset.paths, dataset.targets))
            train_set.paths = [dataset.paths[i] for i in train_idx]
            train_set.targets = [dataset.targets[i] for i in train_idx]
            val_set.paths = [dataset.paths[i] for i in val_idx]
            val_set.targets = [dataset.targets[i] for i in val_idx]
            # val_set.transforms = transforms.Compose([transforms.ToTensor()])

    if args.verbose:
        _logger.info("Training set:\n{}".format(train_set))
        _logger.info("Validation set:\n{}".format(val_set))

    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=train_set,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            dataset=val_set,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=not args.distributed,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.multi:
        train_criterion = nn.BCELoss().cuda(args.gpu)
        val_criterion = nn.BCELoss().cuda(args.gpu)
    else:
        train_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        val_criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    best_metric = None
    for epoch in range(start_epoch, args.epochs):
        train(args, epoch, model, train_loader, optimizer, train_criterion, logger=_logger)
        val_loss, val_acc, val_recal = validate(args, epoch, model, val_loader, val_criterion, logger=_logger)
        scheduler.step(val_loss)
        if best_metric is not None and val_loss < best_metric:
            is_best = True
            best_metric = val_loss
        elif best_metric is None:
            is_best = True
            best_metric = val_loss
        else:
            is_best = False
        if args.verbose:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict() if not args.distributed else model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_checkpoint(checkpoint, args.output, epoch, val_loss, val_acc, is_best)
        dist.barrier()


def train(args, epoch, model, loader, optimizer, criterion, logger):
    steps = len(loader)
    local_loss = AverageMeter()
    local_acc = AverageMeter()
    local_recall = AverageMeter()
    aver_loss = AverageMeter()
    aver_acc = AverageMeter()
    aver_recall = AverageMeter()

    model.train()
    if args.verbose:
        logger.info("Training")

    for i, (images, targets) in enumerate(loader, start=1):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        outputs = model(images)
        if args.multi:
            outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.multi:
            precision, recall = calculate_metrics(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy(),
                                                  args.threshold)
        else:
            precision = accuracy(outputs, targets)[0].item()
            recall = precision

        local_loss.update(loss.item(), images.size(0))
        local_acc.update(precision, images.size(0))
        local_recall.update(recall, images.size(0))

        if args.distributed:
            running_metrics = torch.FloatTensor([loss.item(), precision, recall]).cuda(args.gpu)
            dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
            running_metrics /= args.world_size
            aver_loss.update(running_metrics[0].item())
            aver_acc.update(running_metrics[1].item())
            aver_recall.update(running_metrics[2].item())
        else:
            aver_loss.update(loss.item(), images.size(0))
            aver_acc.update(precision, images.size(0))
            aver_recall.update(recall, images.size(0))

        if args.verbose and i % args.log_interval == 0:
            logger.info(
                "Epoch: [{}] [{}]/[{}]({:.2%}) "
                "Loss: {:.4f} / {:.4f} / {:.4f} "
                "Acc: {:.2f} / {:.2f} / {:.2f} "
                "Recall: {:.2f} / {:.2f} / {:.2f}".format(
                    epoch, i, steps, i / steps, loss, local_loss.avg, aver_loss.avg, precision, local_acc.avg,
                    aver_acc.avg, recall, local_recall.avg, aver_recall.avg))

    return aver_loss.avg, aver_acc.avg, aver_recall.avg


def validate(args, epoch, model, loader, criterion, logger):
    steps = len(loader)
    local_loss = AverageMeter()
    local_acc = AverageMeter()
    local_recall = AverageMeter()
    aver_loss = AverageMeter()
    aver_acc = AverageMeter()
    aver_recall = AverageMeter()

    model.eval()
    if args.verbose:
        logger.info("Validating")

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader, start=1):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

            outputs = model(images)
            if args.multi:
                outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, targets)

            if args.multi:
                precision, recall = calculate_metrics(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy(),
                                                      args.threshold)
            else:
                precision = accuracy(outputs, targets)[0].item()
                recall = precision

            local_loss.update(loss.item(), images.size(0))
            local_acc.update(precision, images.size(0))
            local_recall.update(recall, images.size(0))

            if args.distributed:
                running_metrics = torch.FloatTensor([loss.item(), precision, recall]).cuda(args.gpu)
                running_metrics /= args.world_size
                dist.all_reduce(running_metrics, op=dist.ReduceOp.SUM)
                aver_loss.update(running_metrics[0].item())
                aver_acc.update(running_metrics[1].item())
                aver_recall.update(running_metrics[2].item())
            else:
                aver_loss.update(loss.item(), images.size(0))
                aver_acc.update(precision, images.size(0))
                aver_recall.update(recall, images.size(0))

            if args.verbose and i % args.log_interval == 0:
                logger.info(
                    "Epoch: [{}] [{}]/[{}]({:.2%}) "
                    "Loss: {:.4f} / {:.4f} / {:.4f} "
                    "Acc: {:.2f} / {:.2f} / {:.2f} "
                    "Recall: {:.2f} / {:.2f} / {:.2f}".format(
                        epoch, i, steps, i / steps, loss, local_loss.avg, aver_loss.avg, precision, local_acc.avg,
                        aver_acc.avg, recall, local_recall.avg, aver_recall.avg))

    return aver_loss.avg, aver_acc.avg, aver_recall.avg


if __name__ == '__main__':
    main()
