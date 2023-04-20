from inspect import EndOfBlock
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import join
import re
from .rimednet.dmp import DMP
import sys
import time
import copy
import projects.transforms as transforms
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as tf
from .rimednet.dmp_integrator import DMP_integrate_batch_2

class IntentCollator(object):
    
    def __init__(self, extend=False, image_shape=None):
        self.extend = extend
        self.d = image_shape
    
    def __call__(self, batch):
        lengths, batch_out = {}, {}
        for key in batch[0].keys():
            if key == 'demo':
                batch_out[key] = torch.Tensor([torch.Tensor(t[key]) for t in batch])
            elif 'dmp' in key:
                batch_out[key] = [torch.Tensor(t[key].float()) for t in batch]
                batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                        batch_out[key], batch_first=True)
            elif 'imu' in key:
                batch_out[key] = [torch.Tensor(t[key].float()) for t in batch]
                batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                        batch_out[key], batch_first=True)
            elif 'domain' in key:
                domains = [t[key].unsqueeze(0) for t in batch]
                batch_out[key] = torch.cat(domains, dim=0)
            elif 'video' in key:
                lengths[key] = torch.Tensor([t[key].shape[0] for t in batch])
                max_len = max(lengths[key])
                if self.extend:
                    batch_out[key] = torch.empty((len(lengths[key]),
                        int(max_len), self.d[1], self.d[2], self.d[3]))
                    for i, t in enumerate(batch):
                        batch_out[key][i] = torch.cat((t[key], 
                            t[key][-1].unsqueeze(0).expand(
                            int(max_len-t[key].shape[0]),
                            self.d[1], self.d[2], self.d[3])))
                        lengths[key][i] = max_len
                else:
                    batch_out[key] = torch.zeros((len(lengths[key]),
                        int(max_len), self.d[1], self.d[2], self.d[3]))
                    for i, t in enumerate(batch):
                        batch_out[key][i][0:len(t[key])] = t[key]
            else:
                lengths[key] = torch.Tensor([t[key].shape[0] for t in batch])
                batch_out[key] = [torch.Tensor(t[key]) for t in batch]
                batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                        batch_out[key], batch_first=True)
    
        return batch_out, lengths


class Collator(object):
    
    def __init__(self, extend=False, image_shape=None):
        self.extend = extend
        self.d = image_shape
    
    def __call__(self, batch):
        lengths, old_lengths, batch_out = {}, {}, {}
        for key in ('rgb_video', 'depth_video', 'dmp_gc', 
                     'full_traj_gc', 'train_curve_gc', 'domain'):
            if key == 'demo':
                batch_out[key] = torch.Tensor([torch.Tensor(t[key]) for t in batch])
            elif 'dmp' in key:
                batch_out[key] = [torch.Tensor(t[key].float()) for t in batch]
                batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                        batch_out[key], batch_first=True)
            elif 'imu' in key:
                batch_out[key] = [torch.Tensor(t[key].float()) for t in batch]
                batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                        batch_out[key], batch_first=True)
            elif 'domain' in key:
                domains = [t[key].unsqueeze(0) for t in batch]
                batch_out[key] = torch.cat(domains, dim=0)
            elif 'video' in key:
                lengths[key] = torch.Tensor([t[key].shape[0] for t in batch])
                old_lengths[key] = copy.deepcopy(lengths[key])
                max_len = max(lengths[key])
                if self.extend:
                    batch_out[key] = torch.empty((len(lengths[key]),
                        int(max_len), self.d[1], self.d[2], self.d[3]))
                    for i, t in enumerate(batch):
                        batch_out[key][i] = torch.cat((t[key], 
                            t[key][-1].unsqueeze(0).expand(
                            int(max_len-t[key].shape[0]),
                            self.d[1], self.d[2], self.d[3])))
                        lengths[key][i] = max_len
                else:
                    batch_out[key] = torch.zeros((len(lengths[key]),
                        int(max_len), self.d[1], self.d[2], self.d[3]))
                    for i, t in enumerate(batch):
                        batch_out[key][i][0:len(t[key])] = t[key]
            else:
                lengths[key] = torch.Tensor([t[key].shape[0] for t in batch])
                batch_out[key] = [torch.Tensor(t[key]) for t in batch]
                batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                        batch_out[key], batch_first=True)
    
        return batch_out, lengths, old_lengths


class IMUCollator(object):
    
    def __init__(self, extend=False):
        self.extend = extend
    
    def __call__(self, batch):
        lengths, batch_out = {}, {}
        for item in batch:
            for key in item.keys():
                if key == 'demo':
                    batch_out[key] = torch.Tensor([torch.Tensor(t[key]) for t in batch])
                elif key in {'dmp', 'imu'}:
                    batch_out[key] = [torch.Tensor(t[key].float()) for t in batch]
                    batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                            batch_out[key], batch_first=True)
                elif key == 'traj':
                    batch_out[key] = torch.stack([torch.Tensor(t[key][-1]) for t in batch])
                else:
                    lengths[key] = torch.Tensor([t[key].shape[0] for t in batch])
                    batch_out[key] = [torch.Tensor(t[key]) for t in batch]
                    if self.extend:
                        for i, t in enumerate(batch_out[key]):
                            while len(t) < max(lengths[key]):
                                t = torch.cat((t, t[-1].unsqueeze(0)))
                            batch_out[key][i] = t
                            if 'video' in key:
                                lengths[key][i] = max(lengths[key])
                        batch_out[key] = torch.stack(batch_out[key])
                    else:
                        # if 'video' in key:
                        #     for i, t in enumerate(batch_out[key]):
                        #         batch_out[key][i] = t[:-1]
                        #         lengths[key][i] -= 1
                        batch_out[key] = torch.nn.utils.rnn.pad_sequence(
                            batch_out[key], batch_first=True)
    
        return batch_out, lengths


def natural_sort(l):
    
    """Natural sorting of lists."""
    convert = lambda text: (int(text) if text.isdigit() else text.lower())
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

        
def create_dataloader(dataset, num_workers,
                      batch_size, train_percent, 
                      indices_path, split_seed,
                      model_save_path, collate_fn=None,
                      drop_last=False):
    train_size = []
    val_size = []
    for i, tp in enumerate(train_percent):
        train_size.append(int(tp * len(dataset.datasets[i])))
        val_size.append(len(dataset.datasets[i]) - train_size[i])
    
    train_ds, val_ds = [], []
    train_indices, val_indices = [], []

    for i in range(len(dataset.datasets)):
        train_idx_str = 'train_indices_' + str(i) + '.npy'
        val_idx_str = 'val_indices_' + str(i) + '.npy'
        
        if indices_path is not None:
            # If loading indices was specified, use indices from specified path
            print('Loading data indices for dataset number {} from {}...'
                  .format(i, indices_path[i]))
            train_indices.append(np.load(os.path.join(indices_path[i], 
                train_idx_str)))
            val_indices.append(np.load(os.path.join(indices_path[i],
                val_idx_str)))

            train_ds.append(torch.utils.data.Subset(dataset.datasets[i], train_indices[i]))
            val_ds.append(torch.utils.data.Subset(dataset.datasets[i], val_indices[i]))
        elif split_seed is not None:
            print('Splitting with a manual seed: {}...'.format(split_seed))
            tds, vds = torch.utils.data.random_split(
                dataset.datasets[i],
                [train_size[i], val_size[i]],
                generator=torch.Generator().manual_seed(split_seed))
            train_ds.append(tds)
            val_ds.append(vds)
        else:
            print('Saving new indices for dataset number {}...'.format(i))
            tds, vds = torch.utils.data.random_split(dataset.datasets[i],
                                                    [train_size[i], val_size[i]])
            train_ds.append(tds)
            val_ds.append(vds)
            train_indices.append(train_ds[i].indices)
            val_indices.append(val_ds[i].indices)
        # if not(os.path.exists(model_save_path)):
        #       os.makedirs(model_save_path)
        # np.save(os.path.join(model_save_path, train_idx_str), train_indices[i])
        # np.save(os.path.join(model_save_path, val_idx_str), val_indices[i])
        
    train_ds = torch.utils.data.ConcatDataset(train_ds)
    # train_ds = torch.utils.data.Subset(torch.utils.data.ConcatDataset(train_ds), [0, 1])

    # print("\n#######\nONLY USING THE LAST DATASET FOR VALIDATION\n#######\n")
    # val_ds = val_ds[-1]

    print("\n#######\n VALIDATION DATASETS ARE SET BY RATIOS\n#######\n")
    val_ds = torch.utils.data.ConcatDataset(val_ds)
    
    train_dl, val_dl = (
        DataLoader(train_ds, batch_size=batch_size,
                   shuffle=True,
                   collate_fn=collate_fn,
                   num_workers=num_workers,
                   pin_memory=True,
                   drop_last=drop_last
                   ),
        DataLoader(val_ds, batch_size=batch_size,
                   collate_fn=collate_fn,
                   num_workers=num_workers,
                   pin_memory=True
                   ))
    
    return train_dl, val_dl
