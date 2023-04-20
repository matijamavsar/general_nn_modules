import numpy as np
from scipy.interpolate import interp1d

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


class TrajRandomize(object):
    """Randomize a trajectory."""

    def __init__(self, noise_var, noise_var3d, offset_range, xy_start_range=None):
        self.noise_var = noise_var
        self.noise_var3d = noise_var3d
        # self.xy_start_range = np.array(xy_start_range)
        self.offset_range = offset_range

    def __call__(self, sample):

        if np.random.uniform() < 0.5:
            traj_offset = np.random.uniform(self.offset_range[0], self.offset_range[1])
            for key in {'pos_raw_camera', 'pos_filt_camera', 
                        'pos_ip_camera', 'pos_fip_camera'}:
                ### Randomize the starting point
                # slen = len(sample[key])
                # coef = np.random.uniform(low=7, high=15)
                # f_exp = np.exp(-100*np.arange(0, slen)/coef/slen)
                # xy_start = np.random.uniform(low=self.xy_start_range[:,0],
                #                              high=self.xy_start_range[:,1])
                # xy_offset = sample[key][0,0:2] - xy_start
                
                # for i, y_exp in enumerate(f_exp):
                #     sample[key][i,0:2] = sample[key][i,0:2] + (xy_offset * y_exp)

                ### Randomize the trajectory offset
                sample[key] = sample[key] + traj_offset

        mean = 0
        sigma = (np.random.uniform()*self.noise_var)**0.5
        sigma3d = (np.random.uniform()*self.noise_var3d)**0.5

        if np.random.uniform() < 0.5:
            for key in sample.keys():
                if '_camera' in key:
                    gauss = np.random.normal(mean, sigma3d, sample[key].shape)
                    weights = np.exp(-20*np.arange(0, gauss.shape[0])/2/gauss.shape[0])
                    gauss = gauss * np.expand_dims(weights, 1)
                    sample[key] = sample[key] + torch.tensor(gauss)
                elif 'pos' in key:
                    gauss = np.random.normal(mean, sigma, sample[key].shape)
                    weights = np.exp(-np.arange(0, gauss.shape[0]))
                    gauss = gauss * np.expand_dims(weights, 1)
                    sample[key] = sample[key] + torch.tensor(gauss)

        return sample

class TrajRandomSample(object):
    """Randomly sample trajectories."""
    
    def __init__(self, frame_range, fix_n_jit=None):
        self.frame_range = frame_range
        self.fix_n_jit = fix_n_jit

    def get_indices(self, frame_number, sample_len):
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

        return frame_indices

    def __call__(self, sample):
        sample_len = {'ip': sample['pos_ip'].shape[0],
                      'raw': sample['pos_raw'].shape[0]}

        frame_number = round(np.random.uniform(self.frame_range[0],
                                               self.frame_range[1]))

        # Temporally sub-sample trajectory
        for key in sample_len.keys():
            frame_indices = self.get_indices(frame_number, sample_len[key])

            if 'color_video' in sample.keys():
                if key == 'ip':
                    rgb_len = len(sample['color_video'])
                    traj_len = len(frame_indices)
                    rgb_step = rgb_len / traj_len
                    rgb_indices = np.arange(0, rgb_len, rgb_step).round()
                    sample['color_video'] = sample['color_video'][rgb_indices]
                    sample['color_video'] = sample['color_video'].permute(0, 3, 1, 2).float() / 255.0

            # if 'depth_video' in sample.keys():
            #     if key == 'ip':
            #         sample['depth_video_ip'] = 0 # TODO
            #     elif key == 'raw':
            #         sample['depth_video_raw'] = 0 # TODO
            for skey in sample.keys():
                if key in skey:
                    sample[skey] = sample[skey][frame_indices]

        if 'depth_video' in sample.keys():
            sample['depth_video_ip'] = sample['depth_video_ip'].permute(0, 3, 1, 2).float() / 255.0
            sample['depth_video_raw'] = sample['depth_video_raw'].permute(0, 3, 1, 2).float() / 255.0
        assert sample['pos_raw'].shape[0] == sample['pos_ip'].shape[0]

        return sample
