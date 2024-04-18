# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

utilities

@author: tadahaya
"""
import os
import datetime
import numpy as np
import pandas as pd
import random
import torch
import logging
import matplotlib.pyplot as plt
from sklearn import metrics

from torchinfo import summary

SEP = os.sep

# assist model building
def fix_seed(seed:int=None,fix_gpu:bool=False):
    """ fix seed """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# logger
def init_logger(
    module_name:str, outdir:str='', tag:str='',
    level_console:str='info', level_file:str='info'
    ):
    """
    initialize logger
    
    """
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag)==0:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(
        level=level_dic[level_file],
        filename=f'{outdir}{SEP}log_{tag}.txt',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y%m%d-%H%M%S',
        )
    logger = logging.getLogger(module_name)
    sh = logging.StreamHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
        )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def to_logger(
    logger, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True
    ):
    """ add instance information to logging """
    logger.info(name)
    for k,v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden:
                if not k.startswith('_'):
                    logger.info('  {0}: {1}'.format(k,v))
            else:
                logger.info('  {0}: {1}'.format(k,v))


# save & export
def summarize_model(model, input, outdir):
    """
    summarize model using torchinfo

    Parameters
    ----------
    outdir: str
        output directory path

    model:
        pytorch model
    
    input:
        input tensor
    
    """
    try:
        with open(outdir + SEP + 'summary.txt', 'w') as writer:
            writer.write(repr(summary(model, input.size())))
    except ModuleNotFoundError:
        print('!! CAUTION: no torchinfo and model summary was not saved !!')
    torch.save(model.state_dict(), outdir + SEP + 'model.pt')


def save_loss(train_loss, valid_loss, outdir, name:str=""):
    """ save loss """
    df = pd.DataFrame({"train_loss":train_loss, "valid_loss":valid_loss})
    if len(name) > 0:
        df.to_csv(outdir + SEP + f'loss_curve_{name}.txt', sep='\t')
    else:
        df.to_csv(outdir + SEP + f'loss_curve.txt', sep='\t')


# plot
def plot_progress(
        train_loss, test_loss, num_epoch, outdir, xlabel="epoch", ylabel="loss"
        ):
    """ plot learning progress """
    epochs = list(range(1, num_epoch + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(epochs, train_loss, c='purple', label='train')
    ax.plot(epochs, test_loss, c='orange', label='valid')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + SEP + f'progress_{ylabel}.tif', dpi=100, bbox_inches='tight')


def plot_accuracy(scores, labels, outdir):
    """ plot learning progress """
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auroc = metrics.auc(fpr, tpr)
    precision, _, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(tpr, precision)
    fig, axes = plt.subplots(1, 2, tight_layout=True)
    plt.rcParams['font.size'] = 18
    axes[0, 1].plot(fpr, tpr, c='purple')
    axes[0, 1].set_title(f'ROC curve (area: {auroc:.3})')
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 2].plot(tpr, precision, c='orange')
    axes[0, 2].set_title(f'PR curve (area: {aupr:.3})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    plt.grid()
    plt.savefig(outdir + SEP + 'accuracy.tif', dpi=100, bbox_inches='tight')
    df = pd.DataFrame({'labels':labels, 'predicts':scores})
    df.to_csv(outdir + SEP + 'predicted.txt', sep='\t')
    return auroc, aupr