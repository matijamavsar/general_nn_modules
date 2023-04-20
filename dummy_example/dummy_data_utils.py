import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video
import numpy as np
import os
import re
import sys
import torchvision.transforms.functional as TF


def natural_sort(l):
    """Natural sorting of lists."""
    convert = lambda text: (int(text) if text.isdigit() else text.lower())
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
    

class DummyDataset(Dataset):
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