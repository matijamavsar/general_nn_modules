from os.path import join, exists
from projects.talosnet.data import VideoTrajDataset
import torchvision.transforms as tf
from torchvision.utils import save_image
from pprint import pprint
import multiprocessing as mp
import projects.talosnet.traj_transforms as transforms
import torch
import sys
import os

global dataset

class ArgsClass():
    def __init__(self):
        self.data_load_path = sys.argv[1]
        self.output_dir = sys.argv[2]
        self.depth_video_regex = 'body_depth_video_\d+.avi'
        self.camera_regex = 'body_camera_\d+.npy'
        self.data_regex = 'body_landmarks_\d+.npy'
        self.multiple_runs = 150
        self.subset_size = None
        self.split_seed = 42
args = ArgsClass()

tranlist = tf.Compose([
    transforms.TrajRandomSample([10, 20], 0),

    transforms.TrajRandomize(0.0001, 0.0001, [[-0.2, -0.2, -0.2], [0.2, 0.2, 0.2]]) # low noise
    # transforms.TrajRandomize(0.0005, 0.0003) # high noise
])

print("Will write into dir:", args.output_dir)
print("The following transforms will be used:", 
    [vars(tran) for tran in tranlist.transforms])
input("PRESS ENTER TO CONTINUE")

dataset = VideoTrajDataset(
    data_load_path=args.data_load_path, 
    depth_video_regex=args.depth_video_regex, 
    data_regex=args.data_regex,
    camera_regex=args.camera_regex,
    tranlist=tranlist)

if args.subset_size is not None:
    lengths = [args.subset_size, len(dataset) - args.subset_size]
    dataset = torch.utils.data.random_split(dataset, lengths, 
        generator=torch.Generator().manual_seed(args.split_seed))[0]

if not exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(join(args.output_dir, "info.log"), "w") as log_file:
    pprint(vars(args), log_file)
    for tran in tranlist.transforms:
        pprint(vars(tran), log_file)

def write_sample(idx):
    output_filename = join(args.output_dir, 'sample_' + str(sample_num))
    tmp_sample = dataset[idx]

    if 'depth_video' in tmp_sample.keys():
        save_image(tmp_sample['depth_video'], output_filename + 
            '_dep.png', nrow=1, padding=0)
        del tmp_sample['depth_video']
    torch.save(tmp_sample, output_filename + '_data.pkl')

sample_num = 0

if not exists(args.output_dir):
    os.makedirs(args.output_dir)

for run in range(args.multiple_runs):
    
    # pool = mp.Pool(4)
    # pool.map(write_sample, [idx for idx in range(len(dataset))])
    # pool.close()

    for idx in range(len(dataset)):
        # try:
            write_sample(idx)
            print('Saving sample number', sample_num, end='\r')

            sample_num += 1
        # except Exception as e:
        #     print('\nSkipping sample', idx)
        #     print('\nError:', e)

print('\nFinished transforming data\n')