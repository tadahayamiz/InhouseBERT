# -*- coding: utf-8 -*-
"""

CLI for training and encoding histogram into vector

@author: tadahaya
"""
# packages installed in the current environment
import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

# original packages in src
from .src import utils
from .src import data_handler as dh
from .src.models import VAE
from .src.models import loss_function

parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument(
    'workdir',
    type=str,
    help='working directory that contains the dataset'
    )
parser.add_argument('--note', type=str, help='short note for this running')
parser.add_argument('--train', type=bool, default=True) # 学習ありか否か
parser.add_argument('--seed', type=str, default=222)
parser.add_argument('--num_step', type=int, default=5) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--lr', type=float, default=0.001) # learning rate

args = parser.parse_args()
utils.fix_seed(seed=args.seed, fix_gpu=False) # for seed control

# setup
SEP = os.sep
now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
DIR_NAME = args.workdir + SEP + 'results' + SEP + now # for output
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
LOGGER = utils.init_logger(__name__, DIR_NAME, now, level_console='debug') # for logger
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device


def prepare_data():
    """
    データの読み込み・ローダーの準備を実施
    加工済みのものをdataにおいておくか, argumentで指定したパスから呼び出すなりしてデータを読み込む
    inference用を読み込む際のものも用意しておくと楽
    
    """
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10

    train_trans = transforms.Compose([
        transforms.RandomAffine([0, 30], scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = CIFAR10(root='./data', train=True, download=True, transform=train_trans)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=test_trans)
    train_loader = dh.prep_dataloader(train_set, args.batch_size)
    test_loader = dh.prep_dataloader(test_set, args.batch_size)
    return train_loader, test_loader


def prepare_model():
    """
    model, loss, optimizer, schedulerの準備

    """
    model = VAE(output_dim=10)
    model.to(DEVICE)
    criterion = loss_function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return model, criterion, optimizer


def train_step(model, train_loader, test_loader, criterion, optimizer):
    """
    epoch単位の学習構成, なくとも良い
    
    """
    model.train() # training
    train_batch_loss = []
    train_batch_rl = []
    train_batch_kld = []
    for data_in, data_out in train_loader:
        data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE) # put data on GPU
        optimizer.zero_grad() # reset gradients
        output, mu, logvar = model(data_in) # forward
        loss, rl, kld = criterion(output, data_out, mu, logvar) # calculate loss
        loss.backward() # backpropagation
        optimizer.step() # update parameters
        train_batch_loss.append(loss.item())
        train_batch_rl.append(rl.item())
        train_batch_kld.append(kld.item())
    model.eval() # test (validation)
    test_batch_loss = []
    test_batch_rl = []
    test_batch_kld = []
    with torch.no_grad():
        for data_in, data_out in test_loader:
            data_in, data_out = data_in.to(DEVICE), data_out.to(DEVICE)
            output, mu, logvar = model(data_in)
            loss, rl, kld = criterion(output, data_out, mu, logvar)
            test_batch_loss.append(loss.item())
            test_batch_rl.append(rl.item())
            test_batch_kld.append(kld.item())
    train_loss = (
        np.mean(train_batch_loss), np.mean(train_batch_rl), np.mean(train_batch_kld)
        )
    test_loss = (
        np.mean(test_batch_loss), np.mean(test_batch_rl), np.mean(test_batch_kld)
        )
    return model, train_loss, test_loss


def fit(model, train_loader, test_loader, criterion, optimizer):
    """
    学習
    model, train_loss, test_loss (valid_loss)を返す
    
    """
    train_loss = []
    train_rl = []
    train_kld = []
    test_loss = []
    test_rl = []
    test_kld = []
    for step in trange(args.num_step):
        model, train_step_loss, test_step_loss = train_step(
            model, train_loader, test_loader, criterion, optimizer
            )
        train_loss.append(train_step_loss[0])
        train_rl.append(train_step_loss[1])
        train_kld.append(train_step_loss[2])
        test_loss.append(test_step_loss[0])
        test_rl.append(test_step_loss[1])
        test_kld.append(test_step_loss[2])
        if step % 1000 == 0:
            LOGGER.info(
                f'step: {step} // train_loss: {train_step_loss[0]:.4f} // valid_loss: {test_step_loss[0]:.4f}'
                )
    return model, (train_loss, train_rl, train_kld), (test_loss, test_rl, test_kld)


def main():
    if args.train:
        # training mode
        start = time.time() # for time stamp
        # 1. data prep
        train_loader, test_loader = prepare_data()
        LOGGER.info(
            f'num_training_data: {len(train_loader)}, num_test_data: {len(test_loader)}'
            )
        # 2. model prep
        model, criterion, optimizer = prepare_model()
        # 3. training
        model, train_loss, test_loss = fit(
            model, train_loader, test_loader, criterion, optimizer
            )
        utils.plot_progress(train_loss, test_loss, args.num_step, DIR_NAME)
        utils.summarize_model(model, next(iter(train_loader))[0], DIR_NAME)
        # 4. save results & config
        utils.to_logger(LOGGER, name='argument', obj=args)
        utils.to_logger(LOGGER, name='loss', obj=criterion)
        utils.to_logger(
            LOGGER, name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
            )
        LOGGER.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
    else:
        # inference mode
        # データ読み込みをtestのみに変更などが必要
        pass