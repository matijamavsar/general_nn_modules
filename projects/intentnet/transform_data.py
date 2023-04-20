from projects.data_utils import *
from os.path import join, exists
import torchvision.transforms as tf
from pprint import pprint

class ArgsClass():
    def __init__(self):
        self.data_load_path = '/home/share/Data/HandOptiTrack_fullTrayAll/'
        self.output_dir = 'TR1_RANDOMS_2_TRAIN'
        self.rgb_video_regex = 'hand_rgb_video_\d+.avi'
        self.depth_video_regex = 'hand_dep_video_\d+.avi'
        self.traj_regex = 'hand_opti_pose_\d+.npy'
        self.demo_regex = 'hand_demo_label_\d+.npy'
        self.subset_size = None
        self.split_seed = 42
args = ArgsClass()

tranlist = tf.Compose([
    transforms.RandomSample([4, 16], 0), # for train and val
    # transforms.SubSample(4), # for test

    transforms.Transpose(),

    transforms.RandomTransforms([120, 160, -1], 3, -1, 0.3, 0.0007, 0.5), # for randoms
    # transforms.RandomTransforms([120, 160], 3, 0.2, 0.3, 0.0015, 0.5), # for highrandoms
    # transforms.RandomTransforms([120, 160, -1], 1, -1, 0.1, 0.0005), # for val
    # transforms.RandomTransforms([120, 160, -1], 0, -1, 0, 0), # for test

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = VideoTrajDataset(
    data_load_path=args.data_load_path, 
    rgb_video_regex=args.rgb_video_regex, 
    depth_video_regex=args.depth_video_regex, 
    traj_regex=args.traj_regex, 
    demo_regex=args.demo_regex,
    tranlist=tranlist)

if args.subset_size is not None:
    lengths = [args.subset_size, len(dataset) - args.subset_size]
    dataset = torch.utils.data.random_split(dataset, lengths, 
        generator=torch.Generator().manual_seed(args.split_seed))[0]

if not exists(join(args.data_load_path, args.output_dir)):
    os.makedirs(join(args.data_load_path, args.output_dir))

with open(join(args.data_load_path, args.output_dir, "info.log"), "w") as log_file:
    pprint(vars(args), log_file)
    for tran in tranlist.transforms:
        pprint(vars(tran), log_file)

for i in range(len(dataset)):
    output_filename = join(args.data_load_path, args.output_dir, 'sample_' + str(i) + '.pkl')
    torch.save(dataset[i], output_filename)
    print('Saving sample number', i, end='\r')