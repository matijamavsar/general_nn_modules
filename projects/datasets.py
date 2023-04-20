from inspect import EndOfBlock
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_video, read_image
from torch.utils.data import DataLoader
import numpy as np
import os
from os.path import join
import re
from .rimednet.dmp import DMP
import sys
import projects.transforms as transforms
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as tf
from .data_utils import *


class VideoTrajDataset(Dataset):
    
    def __init__(self, data_load_path,
                 rgb_video_regex,
                 depth_video_regex,
                 traj_regex,
                 demo_regex,
                 tranlist):
        self.root_dir = data_load_path
        self.tranlist = tranlist
        file_list = os.listdir(self.root_dir)
    
        rgb_video_r = re.compile(rgb_video_regex)
        depth_video_r = re.compile(depth_video_regex)
        traj_r = re.compile(traj_regex)
        demo_r = re.compile(demo_regex)
        
        rgb_video_files = natural_sort(
            list(filter(rgb_video_r.match, file_list)))
        depth_video_files = natural_sort(
            list(filter(depth_video_r.match, file_list)))
        traj_files = natural_sort(
            list(filter(traj_r.match, file_list)))
        demo_files = natural_sort(
            list(filter(demo_r.match, file_list)))
    
        self.keys = ['rgb_video', 'depth_video', 'traj', 'demo']
        self.sample_files = (rgb_video_files, depth_video_files,
                                traj_files, demo_files)
        
    def __len__(self):
        return len(self.sample_files[0])
    
    def __getitem__(self, idx):
        
        sample = {}
        
        rgb_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('rgb_video')][idx])
        depth_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('depth_video')][idx])
        traj_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('traj')][idx])
        demo_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('demo')][idx])
        
        sample['rgb_video'] = read_video(rgb_video_file, pts_unit='sec')[0]
        sample['depth_video'] = read_video(depth_video_file, pts_unit='sec')[0]
        sample['traj'] = pd.DataFrame(data = np.load(traj_file),
                                      columns = ["x", "y", "z", 
                                                "qx", "qy", "qz", "qw"])
        _, sample['demo'] = torch.tensor(
                            np.load(demo_file)).max(dim=0)
        sample['demo'] = sample['demo'].unsqueeze(0).float()
        sample = self.tranlist(sample) 
        
        return sample


class ImuVidTrajDataset(Dataset):
    
    def __init__(self, data_load_path,
                 rgb_video_regex,
                 depth_video_regex,
                 traj_regex,
                 imu_regex,
                 tranlist):
        self.root_dir = data_load_path
        self.tranlist = tranlist
        file_list = os.listdir(self.root_dir)
    
        rgb_video_r = re.compile(rgb_video_regex)
        depth_video_r = re.compile(depth_video_regex)
        traj_r = re.compile(traj_regex)
        imu_r = re.compile(imu_regex)
        
        rgb_video_files = natural_sort(
            list(filter(rgb_video_r.match, file_list)))
        depth_video_files = natural_sort(
            list(filter(depth_video_r.match, file_list)))
        traj_files = natural_sort(
            list(filter(traj_r.match, file_list)))
        imu_files = natural_sort(
            list(filter(imu_r.match, file_list)))
    
        self.keys = ['rgb_video', 'depth_video', 'traj', 'imu']
        self.sample_files = (rgb_video_files, depth_video_files,
                                traj_files, imu_files)
        
    def __len__(self):
        return len(self.sample_files[0])
    
    def __getitem__(self, idx):
        
        sample = {}
        
        rgb_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('rgb_video')][idx])
        depth_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('depth_video')][idx])
        traj_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('traj')][idx])
        imu_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('imu')][idx])
        
        sample['rgb_video'] = read_video(rgb_video_file, pts_unit='sec')[0]
        sample['depth_video'] = read_video(depth_video_file, pts_unit='sec')[0]
        sample['traj'] = pd.DataFrame(data = np.load(traj_file),
                                      columns = ["x", "y", "z", 
                                                "qx", "qy", "qz", "qw"])
        sample['imu'] = torch.tensor(np.load(imu_file))

        sample = self.tranlist(sample)
        
        return sample

    
class PickleDataset(Dataset):
    
    def __init__(self, data_load_path,
                 pytorch_pickle_regex):
        self.root_dir = data_load_path
        file_list = os.listdir(self.root_dir)
    
        pytorch_pickle_r = re.compile(pytorch_pickle_regex)
        pytorch_pickle_files = natural_sort(
            list(filter(pytorch_pickle_r.match, file_list)))
        self.sample_files = pytorch_pickle_files
        
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):        
        sample = torch.load(os.path.join(
            self.root_dir, self.sample_files[idx]))
        return sample


class PicklePNGDataset(Dataset):
    
    def __init__(self, data_load_path,
                 pytorch_pickle_rgb_regex,
                 pytorch_pickle_dep_regex,
                 pytorch_pickle_data_regex,
                 tranlist=None):
        self.root_dir = data_load_path
        try:
            self.tranlist = tf.Compose([eval(tran) for tran in tranlist])
        except:
            self.tranlist = tranlist
        file_list = os.listdir(self.root_dir)
    
        pytorch_pickle_rgb_r = re.compile(pytorch_pickle_rgb_regex)
        pytorch_pickle_dep_r = re.compile(pytorch_pickle_dep_regex)
        pytorch_pickle_data_r = re.compile(pytorch_pickle_data_regex)
        self.pickle_rgb_files = natural_sort(
            list(filter(pytorch_pickle_rgb_r.match, file_list)))
        self.pickle_dep_files = natural_sort(
            list(filter(pytorch_pickle_dep_r.match, file_list)))
        self.pickle_data_files = natural_sort(
            list(filter(pytorch_pickle_data_r.match, file_list)))

    def __len__(self):
        return len(self.pickle_rgb_files)
    
    def __getitem__(self, idx):
        sample = torch.load(os.path.join(
            self.root_dir, self.pickle_data_files[idx]))
        sample['rgb_video'] = read_image(join(self.root_dir,
            self.pickle_rgb_files[idx]))
        sample['rgb_video'] = sample['rgb_video'].reshape(
            3, -1, sample['rgb_video'].shape[-1], sample['rgb_video'].shape[-1])
        sample['rgb_video'] = sample['rgb_video'].permute(1,0,2,3).float()/255
        sample['depth_video'] = read_image(join(self.root_dir,
            self.pickle_dep_files[idx]))
        sample['depth_video'] = sample['depth_video'].reshape(
            3, -1, sample['depth_video'].shape[-1], sample['depth_video'].shape[-1])
        sample['depth_video'] = sample['depth_video'].permute(1,0,2,3).float()/255

        if self.tranlist is not None:
            sample = self.tranlist(sample)
        return sample


class ImagePickleDataset(Dataset):
    
    def __init__(self, data_load_path,
                 pytorch_pickle_regex,
                 video_length):
        self.root_dir = data_load_path
        self.video_length = video_length
        file_list = os.listdir(self.root_dir)
    
        pytorch_pickle_r = re.compile(pytorch_pickle_regex)
        pytorch_pickle_files = natural_sort(
            list(filter(pytorch_pickle_r.match, file_list)))
        self.sample_files = pytorch_pickle_files
        
    def __len__(self):
        return len(self.sample_files)*self.video_length
    
    def __getitem__(self, idx):
        vid_num = int(idx/self.video_length)
        frame_num = idx % self.video_length
        sample = torch.load(os.path.join(
            self.root_dir, self.sample_files[vid_num]))
        sample['rgb_video'] = sample['rgb_video'][frame_num]
        sample['depth_video'] = sample['depth_video'][frame_num]
        sample['traj'] = sample['traj'][frame_num]
        del sample['demo']
        
        return sample