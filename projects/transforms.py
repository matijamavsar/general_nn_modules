import numpy as np
from scipy.interpolate import interp1d

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class SubSample(object):
    def __init__(self, sub_hz, fix_speed=None, equalize=False):
        self.sub_hz = sub_hz
        self.fix_speed = fix_speed
        self.equalize = equalize
        print('No traj subsampling, converting to Numpy only!')

    def __call__(self, sample):
        sample_len = sample['rgb_video'].shape[0]
        if self.fix_speed is not None:
            sub_hz = self.sub_hz / self.fix_speed
            old_traj_len = len(sample['full_traj_gc'])
            new_traj_time = self.fix_speed * sample_len / 30.0
            new_traj_len = round(new_traj_time * 120.0)
            xold = np.arange(0, old_traj_len)
            xnew = np.linspace(0, old_traj_len-1, new_traj_len)
            frame_indices = np.arange(0, sample_len, sub_hz).round()
            if frame_indices[-1] == sample_len:
                frame_indices[-1] -= 1
        elif self.equalize:
            try:
                traj_time = len(sample['full_traj_gc']) / 120.0
            except Exception:
                traj_time = len(sample['traj']) / 120.0

            vid_time = sample_len / 30.0
            speed_ratio = traj_time / vid_time
            sub_hz = self.sub_hz / speed_ratio
            frame_indices = np.arange(0, sample_len, sub_hz).round()
            if frame_indices[-1] == sample_len:
                frame_indices[-1] -= 1
        else:
            frame_indices = np.arange(0, sample_len, self.sub_hz)
        
        # Temporally sub-sample video
        for key in sample.keys():
            if 'video' in key:
                last_frame = sample[key][-1].unsqueeze(0)
                sample[key] = sample[key][frame_indices]
                sample[key] =  torch.cat((sample[key], last_frame), 0)

            # Convert (and resample if defined)
            elif 'traj' in key:
                try:
                    sample[key] = sample[key].to_numpy()
                except Exception:
                    pass
                if self.fix_speed is not None:
                    print('Interpolating trajectories!')
                    sample[key] = interp1d(xold, sample[key], axis=0)(xnew)

            # Subsample IMU data
            elif 'imu' in key:
                sample[key] = sample[key][frame_indices]
                
        assert(sample['rgb_video'].shape[-1] == 3)
        assert(sample['depth_video'].shape[-1] == 3)

        return sample

                
class RandomSample(object):
    """Randomly sample the videos in a sample."""
    
    def __init__(self, frame_range, fix_n_jit=None):
        self.frame_range = frame_range
        self.fix_n_jit = fix_n_jit

    def __call__(self, sample):
        sample_len = sample['rgb_video'].shape[0]
        frame_number = round(np.random.uniform(self.frame_range[0],
                                               self.frame_range[1]))
        frame_indices = []
        frame_step = int(np.ceil((sample_len) / (frame_number)))
        if self.fix_n_jit is not None:
            frame_step = sample_len / (frame_number - 1)
            sampling_frame = 0
            i = 0
            while i < sample_len:
                if i >= sampling_frame:
                    frame_jit = int(np.random.uniform(-self.fix_n_jit, self.fix_n_jit))
                    frame_indices.append(round(i + frame_jit))
                    sampling_frame += frame_step
                i += 1

        else:
            for i in range(0, sample_len, frame_step):
                frame_indices.append(int(np.random.uniform(i, i + frame_step)))

        if frame_indices[0] < 0:
            frame_indices[0] = 0
        if frame_indices[-1] >= sample_len:
            frame_indices[-1] = sample_len - 1
        
        # Append last frame in any case
        frame_indices.append(sample_len - 1)
        frame_indices = np.array(frame_indices).round().astype(int)
        
        # Temporally sub-sample video
        for key in sample.keys():
            if 'video' in key:
                sample[key] = sample[key][frame_indices]

            # Subsample trajectories
            elif 'traj' in key:
                traj_len = sample[key].shape[0]
                rel_indices = frame_indices / sample_len
                traj_indices = (rel_indices * traj_len).round().astype(int)
                try:
                    sample[key] = sample[key].to_numpy()[traj_indices]
                except Exception:
                    sample[key] = sample[key][traj_indices]

            # Subsample IMU data
            elif 'imu' in key:
                sample[key] = sample[key][frame_indices]

        assert(sample['rgb_video'].shape[-1] == 3)
        assert(sample['depth_video'].shape[-1] == 3)

        return sample


class Transpose(object):
    
    def __call__(self, sample):
        if sample['rgb_video'].shape[-1] == 3:
            sample['rgb_video'] = sample['rgb_video'].permute(0, 3, 1, 2).float() / 255.0
            sample['depth_video'] = sample['depth_video'].permute(0, 3, 1, 2).float() / 255.0
        
        return sample
    

class Normalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        for key in sample.keys():
            if 'video' in key:
                sample[key] = TF.normalize(sample[key], mean=self.mean, std=self.std)
        return sample


class RandomTransforms(object):
    """Apply several transforms with random parameters."""

    def __init__(self,
                 size=None,
                 degrees=None,
                 distortion=None,
                 jitter=None,
                 noise_var=None,
                 hue_var=0.0,
                 flip=None):
        self.size = size
        self.degrees = degrees
        self.distortion = distortion
        self.jitter = jitter
        self.noise_var = noise_var
        self.hue_var = hue_var
        self.flip = flip

    def __call__(self, sample):
        dummy_img = sample['rgb_video']
        length, row = dummy_img.shape[0: 2]

        deg = np.random.uniform(-self.degrees, self.degrees)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            dummy_img[0], [0.75, 1.0], [0.75, 1.34])
        if self.distortion != -1:
            startpoints, endpoints = transforms.RandomPerspective.get_params(
                dummy_img.shape[3], dummy_img.shape[2], self.distortion)
        per_prob = np.random.uniform()
        flip_prob = np.random.uniform()
        br = np.random.uniform(1-self.jitter, 1+self.jitter)
        co = np.random.uniform(1-self.jitter, 1+self.jitter)
        sa = np.random.uniform(1-self.jitter, 1+self.jitter)
        hu = np.random.uniform(-self.hue_var, self.hue_var)
        tran_jit = transforms.ColorJitter((br, br), (co, co), (sa, sa), (hu, hu))

        mean = 0
        sigma = (np.random.uniform()*self.noise_var)**0.5

        # Apply transforms to frames
        for key in sample.keys():
            if 'video' in key:
                new_video = torch.zeros(length, row, self.size[0], self.size[1])
                for k, frame in enumerate(sample[key]):
                    frame = TF.affine(frame, deg, [0, 0], 1, 0)
                    if per_prob > 0.5 and self.distortion != -1:
                        frame = TF.perspective(frame, startpoints, endpoints)
                    if -1 in self.size:
                        frame = TF.resize(frame, self.size[0:2])
                    else:
                        frame = TF.resized_crop(frame, i, j, h, w, self.size)
                    if flip_prob > 0.5 and self.flip:
                        frame = TF.hflip(frame)
                    frame = tran_jit(frame)
                    
                    # Add noise to frame
                    gauss = np.random.normal(mean, sigma, self.size[0:2])
                    gauss = gauss.reshape(self.size[0:2])
                    new_video[k] = np.clip(frame + gauss, 0, 1)
                    
                sample[key] = new_video

        return sample