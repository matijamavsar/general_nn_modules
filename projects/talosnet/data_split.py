import numpy as np
import sys
import os
import torch
import shutil
from sklearn.model_selection import train_test_split

data_path = sys.argv[1]

data_raw = []
data_filt = []
data_ip = []
data_fip = []

for file in os.listdir(data_path):
    if '_data.pkl' in file:
        data = torch.load(os.path.join(data_path, file))
        data_raw.append(data['pos_raw_camera'])
        data_filt.append(data['pos_filt_camera'])
        data_ip.append(data['pos_ip_camera'])
        data_fip.append(data['pos_fip_camera'])

(data_raw_train, data_raw_test,
data_filt_train, data_filt_test, 
data_ip_train, data_ip_test, 
data_fip_train, data_fip_test) = train_test_split(
    data_raw, data_filt, data_ip, data_fip, test_size=0.15, random_state=42)

(data_raw_train, data_raw_val,
data_filt_train, data_filt_val, 
data_ip_train, data_ip_val, 
data_fip_train, data_fip_val) = train_test_split(
    data_raw_train, data_filt_train, data_ip_train, data_fip_train, test_size=0.21, random_state=42)

sample = {'pos_raw_camera': [], 'pos_filt_camera': [], 'pos_ip_camera': [], 'pos_fip_camera': []}

if os.path.exists(os.path.join(data_path, 'train')):
    shutil.rmtree(os.path.join(data_path, 'train'))
if os.path.exists(os.path.join(data_path, 'val')):
    shutil.rmtree(os.path.join(data_path, 'val'))
if os.path.exists(os.path.join(data_path, 'test')):
    shutil.rmtree(os.path.join(data_path, 'test'))

os.makedirs(os.path.join(data_path, 'train'))
os.makedirs(os.path.join(data_path, 'val'))
os.makedirs(os.path.join(data_path, 'test'))

for idx, (sample_raw_train, sample_filt_train, sample_ip_train, sample_fip_train) in enumerate(zip(
    data_raw_train, data_filt_train, data_ip_train, data_fip_train)):
    sample['pos_raw_camera'] = sample_raw_train
    sample['pos_filt_camera'] = sample_filt_train
    sample['pos_ip_camera'] = sample_ip_train
    sample['pos_fip_camera'] = sample_fip_train

    torch.save(sample, os.path.join(data_path, 'train', 'sample_' + str(idx) + '_data.pkl'))

for idx, (sample_raw_val, sample_filt_val, sample_ip_val, sample_fip_val) in enumerate(zip(
    data_raw_val, data_filt_val, data_ip_val, data_fip_val)):
    sample['pos_raw_camera'] = sample_raw_val
    sample['pos_filt_camera'] = sample_filt_val
    sample['pos_ip_camera'] = sample_ip_val
    sample['pos_fip_camera'] = sample_fip_val

    torch.save(sample, os.path.join(data_path, 'val', 'sample_' + str(idx) + '_data.pkl'))

for idx, (sample_raw_test, sample_filt_test, sample_ip_test, sample_fip_test) in enumerate(zip(
    data_raw_test, data_filt_test, data_ip_test, data_fip_test)):
    sample['pos_raw_camera'] = sample_raw_test
    sample['pos_filt_camera'] = sample_filt_test
    sample['pos_ip_camera'] = sample_ip_test
    sample['pos_fip_camera'] = sample_fip_test

    torch.save(sample, os.path.join(data_path, 'test', 'sample_' + str(idx) + '_data.pkl'))