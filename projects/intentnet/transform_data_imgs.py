from projects.data_utils import *
from projects.datasets import *
from os.path import join, exists
import torchvision.transforms as tf
from torchvision.utils import save_image
from pprint import pprint
import multiprocessing as mp

global dataset

class ArgsClass():
    def __init__(self):
        self.data_load_path = sys.argv[1]
        self.output_dir = sys.argv[2]
        self.rgb_video_regex = 'hand_rgb_video_\d+.avi'
        self.depth_video_regex = 'hand_dep_video_\d+.avi'
        self.traj_regex = 'hand_opti_pose_\d+.npy'
        self.demo_regex = 'hand_demo_label_\d+.npy'
        self.subset_size = None
        self.split_seed = 42
args = ArgsClass()

tranlist = tf.Compose([
    # transforms.RandomSample([4, 16], 0) # not recommended

    # transforms.SubSample(7), # for sim good speed vids
    # transforms.SubSample(7, equalize=True), # for real slow speed vids (fix speed)

    transforms.Transpose(),

    # transforms.RandomTransforms([227, 227], 4, -1, 0.3, 0.0007, 0.5), # for sim data
    # transforms.RandomTransforms([228, 228, -1], 3, -1, 0.3, 0.0007, 0.3), # for real randoms
    # transforms.RandomTransforms([227, 227], 3, 0.2, 0.3, 0.0015, 0.5), # for highrandoms
    # transforms.RandomTransforms([228, 228, -1], 1, -1, 0.1, 0.0005, 0.1), # for val
    transforms.RandomTransforms([228, 228, -1], 0, -1, 0, 0), # for test

    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

print("Will write into dir:", args.output_dir)
print("The following transforms will be used:", 
    [vars(tran) for tran in tranlist.transforms])
# input("PRESS ENTER TO CONTINUE")

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

if not exists(args.output_dir):
    os.makedirs(args.output_dir)

with open(join(args.output_dir, "info.log"), "w") as log_file:
    pprint(vars(args), log_file)
    for tran in tranlist.transforms:
        pprint(vars(tran), log_file)

def write_sample(idx):
    output_filename = join(args.output_dir, 'sample_' + str(idx))
    if (exists(output_filename + '_rgb.png') and 
        exists(output_filename + '_dep.png') and
        exists(output_filename + '_data.pkl')):
        print('Skipping sample number', idx)
        return

    # try:
    tmp_sample = dataset[idx]
    # except Exception as e:
    #     print('Error when reading file number', idx, '- skipping')
    #     print('Error message:', e)
    #     return
    save_image(tmp_sample['rgb_video'], output_filename + 
        '_rgb.png', nrow=1, padding=0)
    save_image(tmp_sample['depth_video'], output_filename + 
        '_dep.png', nrow=1, padding=0)
    del tmp_sample['rgb_video']
    del tmp_sample['depth_video']
    torch.save(tmp_sample, output_filename + '_data.pkl')
    print('Saving sample number', idx, end='\r')

pool = mp.Pool(4)
pool.map(write_sample, [idx for idx in range(len(dataset))])
pool.close()

# for idx in range(len(dataset)):
#     write_sample(idx)

print("\nFinished transforming data\n")
