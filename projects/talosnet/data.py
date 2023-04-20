import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torchvision.io import read_video, read_image
from ..data_utils import create_dataloader, Collator, natural_sort
import copy
import os
import re
import numpy as np

class MediaCollator(object):
    
    def __init__(self, extend=True):
        self.extend = extend
    
    def __call__(self, batch):
        lengths, batch_out, old_lengths = {}, {}, {}
        for key in batch[0].keys():
            if key == 'slot':
                batch_out[key] = torch.cat([t[key].unsqueeze(0) for t in batch])
            elif 'pos' in key:
                lengths[key] = torch.Tensor([t[key].shape[0] for t in batch])
                old_lengths[key] = copy.deepcopy(lengths[key])
                max_len = max(lengths[key])
                if self.extend:
                    batch_out[key] = torch.empty((len(lengths[key]),
                        int(max_len), 3))
                    for i, t in enumerate(batch):
                        batch_out[key][i] = torch.cat((t[key], 
                            t[key][-1].unsqueeze(0).expand(
                            int(max_len-t[key].shape[0]), 3)))
                        lengths[key][i] = max_len
                else:
                    batch_out[key] = torch.zeros((len(lengths[key]),
                        int(max_len), self.d[1], self.d[2], self.d[3]))
                    for i, t in enumerate(batch):
                        batch_out[key][i][0:len(t[key])] = t[key]
    
        return batch_out, lengths, old_lengths


class VideoTrajDataset(Dataset):
    
    def __init__(self, data_load_path,
                 depth_video_regex,
                 data_regex, camera_regex,
                 tranlist):
        self.root_dir = data_load_path
        self.tranlist = tranlist
        file_list = os.listdir(self.root_dir)
    
        depth_r = re.compile(depth_video_regex)
        data_r = re.compile(data_regex)
        camera_r = re.compile(camera_regex)
        
        depth_video_files = natural_sort(
            list(filter(depth_r.match, file_list)))
        data_files = natural_sort(
            list(filter(data_r.match, file_list)))
        camera_files = natural_sort(
            list(filter(camera_r.match, file_list)))
    
        self.keys = ['depth_video', 'data', 'camera']
        self.sample_files = (depth_video_files,
                             data_files, camera_files)
        
    def __len__(self):
        return len(self.sample_files[0])
    
    def __getitem__(self, idx):
        
        sample = {}
        
        depth_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('depth_video')][idx])
        data_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('data')][idx])
        camera_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('camera')][idx])
        
        # sample['depth_video'] = read_video(depth_video_file, pts_unit='sec')[0]
        sample['pos_raw'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_raw'])
        sample['pos_filt'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_filt'])
        sample['pos_ip'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_ip'])
        sample['pos_fip'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_fip'])
        sample['pos_raw_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_raw_camera'])
        sample['pos_filt_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_filt_camera'])
        sample['pos_ip_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_ip_camera'])
        sample['pos_fip_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_fip_camera'])

        sample = self.tranlist(sample)
        
        return sample


class PlotVideoTrajDataset(Dataset):
    
    def __init__(self, data_load_path,
                 color_video_regex,
                 depth_video_regex,
                 data_regex, camera_regex,
                 tranlist):
        self.root_dir = data_load_path
        self.tranlist = tranlist
        file_list = os.listdir(self.root_dir)
    
        color_r = re.compile(color_video_regex)
        depth_r = re.compile(depth_video_regex)
        data_r = re.compile(data_regex)
        camera_r = re.compile(camera_regex)
        
        color_video_files = natural_sort(
            list(filter(color_r.match, file_list)))
        depth_video_files = natural_sort(
            list(filter(depth_r.match, file_list)))
        data_files = natural_sort(
            list(filter(data_r.match, file_list)))
        camera_files = natural_sort(
            list(filter(camera_r.match, file_list)))

        self.keys = ['color_video', 'depth_video', 'data', 'camera']
        self.sample_files = (color_video_files, depth_video_files,
                             data_files, camera_files)
        
    def __len__(self):
        return len(self.sample_files[0])
    
    def __getitem__(self, idx):
        
        sample = {}
        
        color_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('color_video')][idx])
        depth_video_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('depth_video')][idx])
        data_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('data')][idx])
        camera_file = os.path.join(
            self.root_dir, self.sample_files
            [self.keys.index('camera')][idx])
        
        # sample['depth_video'] = read_video(depth_video_file, pts_unit='sec')[0]
        sample['color_video'] = read_video(color_video_file, pts_unit='sec')[0]
        sample['pos_raw'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_raw'])
        sample['pos_filt'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_filt'])
        sample['pos_ip'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_ip'])
        sample['pos_fip'] = torch.tensor(np.load(data_file, allow_pickle=True).tolist()['pos_fip'])
        sample['pos_raw_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_raw_camera'])
        sample['pos_filt_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_filt_camera'])
        sample['pos_ip_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_ip_camera'])
        sample['pos_fip_camera'] = torch.tensor(np.load(camera_file, allow_pickle=True).tolist()['pos_fip_camera'])

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
                 pytorch_pickle_dep_regex,
                 pytorch_pickle_data_regex,
                 tranlist=None):
        self.root_dir = data_load_path
        file_list = os.listdir(self.root_dir)
    
        pytorch_pickle_dep_r = re.compile(pytorch_pickle_dep_regex)
        pytorch_pickle_data_r = re.compile(pytorch_pickle_data_regex)
        self.pickle_dep_files = natural_sort(
            list(filter(pytorch_pickle_dep_r.match, file_list)))
        self.pickle_data_files = natural_sort(
            list(filter(pytorch_pickle_data_r.match, file_list)))

    def __len__(self):
        return len(self.pickle_dep_files)
    
    def __getitem__(self, idx):
        sample = torch.load(os.path.join(
            self.root_dir, self.pickle_data_files[idx]))
        sample['depth_video'] = read_image(os.path.join(self.root_dir,
            self.pickle_dep_files[idx]))
        sample['depth_video'] = sample['depth_video'].reshape(
            3, -1, sample['depth_video'].shape[-1], sample['depth_video'].shape[-1])
        sample['depth_video'] = sample['depth_video'].permute(1,0,2,3).float()/255

        return sample


class TalosData(LightningDataModule):
    def __init__(self, args):

        dataset = []

        for i, data_path in enumerate(args.data_load_path):
            dataset_pick = PickleDataset(
                data_load_path=data_path,
                pytorch_pickle_regex=args.pytorch_pickle_regex,
            )

            len_1 = int(args.data_percent[i] * len(dataset_pick))
            len_2 = len(dataset_pick) - len_1
            dataset_pick, _ = torch.utils.data.random_split(dataset_pick,
                [len_1, len_2])
            print('Loaded dataset from', data_path, 'with length', len(dataset_pick))
            dataset.append(dataset_pick)

        dataset = ConcatDataset(dataset)
        collator = MediaCollator(args.extend)

        self.train_dl, self.val_dl = create_dataloader(
            dataset=dataset,
            model_save_path=args.model_save_path,
            indices_path=args.indices_path,
            split_seed=args.split_seed,
            batch_size=args.batch_size,
            train_percent=args.train_percent,
            num_workers=args.num_workers,
            collate_fn=collator)

        test_ds = PickleDataset(
            data_load_path=args.test_set,
            pytorch_pickle_regex=args.pytorch_pickle_regex
        )

        self.test_dl = DataLoader(test_ds, 
                                  batch_size=args.batch_size,
                                  collate_fn=collator,
                                  num_workers=args.num_workers,
                                  pin_memory=True),
        
    def prepare_data(self):
        0

    def setup(self):
        0
        
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl
    
    def teardown(self):
        0