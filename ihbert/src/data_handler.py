# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataloader

@author: tadahaya
"""
import time
from tqdm import tqdm
import numpy as np
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(
            self, corpus_path, vocab, seq_len, encoding="utf-8",
            corpus_lines=None, on_memory=True
            ):
        self.vocab = vocab # vocab instance
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_path = corpus_path
        self.corpus_lines = corpus_lines
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm(f, desc="Loading Dataset", unit=" lines"):
                    self.corpus_lines += 1
            if on_memory:
                # on memory mode: load all corpus
                self.lines = [line[:-1].split("\t")
                              for line in tqdm(
                                  f, desc="Loading Dataset", unit=" lines",
                                  )]
                self.corpus_lines = len(self.lines)
        if not on_memory:
            # on-the-fly mode
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)
            for _ in range(random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
                # iterate to a random line


    def __len__(self):
        return self.corpus_lines
    

    def __getitem__(self, item):
        # here, we do not use next sentence prediction
        # get a sentence from the corpus
        t = self.get_corpus_line(item)
        # mask random words
        t_random, t_label = self.random_word(t)
        # add sos and eos tokens
        t = [self.vocab.sos_index] + t_random + [self.vocab.eos_index]
        # prepare labels for bert
        t_label = [self.vocab.pad_index] + t_label + [self.vocab.pad_index]
        # summarize bert inputs
        bert_input = t[:self.seq_len]
        bert_label = t_label[:self.seq_len]
        # padding
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        # convert to tensor
        output = {"bert_input": bert_input, "bert_label": bert_label}
        return {key: torch.tensor(value) for key, value in output.items()}


    def get_corpus_line(self, item):
        """ get line from corpus """
        if self.on_memory:
            return self.lines[item]
        else:
            line = self.file.readline()
            if line == "":
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.readline()
            return line[:-1].split("\t")


    def random_word(self, sentence):
        """ get word masked """
        tokens = sentence.split()
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))
                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                    # 何をやっているか
                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(0)
        return tokens, output_label
    




# frozen
class MyDataset(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, input=None, output=None, transform=None):
        if input is None:
            raise ValueError('!! Give input !!')
        if output is None:
            raise ValueError('!! Give output !!')
        if type(transform) == list:
            if len(transform) != 0:
                if transform[0] is None:
                    self.transform = []
                else:
                    self.transform = transform
            else:
                self.transform = transform
        else:
            if transform is None:
                self.transform = []
            else:
                self.transform = [transform]
        self.input = input
        self.output = output
        self.datanum = len(self.input)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        input = self.input[idx]
        output = self.output[idx]
        if len(self.transform) > 0:
            for t in self.transform:
                input = t(input)
        return input, output


class MyTransforms:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.float) -> torch.Tensor:
        x = torch.from_numpy(x.astype(np.float32))  # example
        return x


def prep_dataset(input, output, transform=None) -> torch.utils.data.Dataset:
    """
    prepare dataset from row data
    
    Parameters
    ----------
    data: array
        input data such as np.array

    label: array
        input labels such as np.array
        would be None with unsupervised learning

    transform: a list of transform functions
        each function should return torch.tensor by __call__ method
    
    """
    return MyDataset(input, output, transform)


def prep_dataloader(
    dataset, batch_size, shuffle=None, num_workers=2, pin_memory=True
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn
        )    
    return loader


def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def prep_data(
    train_x, train_y, test_x, test_y, batch_size,
    transform=(None, None), shuffle=(True, False),
    num_workers=2, pin_memory=True
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    combination of prep_dataset and prep_dataloader for model building
    
    Parameters
    ----------
    train_x, train_y, test_x, test_y: arrays
        arrays for training data, training labels, test data, and test labels
    
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing    

    """
    train_dataset = prep_dataset(train_x, train_y, transform[0])
    test_dataset = prep_dataset(test_x, test_y, transform[1])
    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )
    return train_loader, test_loader