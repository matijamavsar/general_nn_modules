import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from ..data_utils import create_dataloader, IntentCollator
from ..datasets import PickleDataset
import os

class VideoTrajDataset(LightningDataModule):
    def __init__(self, args):

        dataset = []

        for data_path in args.data_load_path:
            dataset.append(PickleDataset(
                data_load_path=data_path,
                pytorch_pickle_regex=args.pytorch_pickle_regex))
        dataset = torch.utils.data.ConcatDataset(dataset)

        image_shape = dataset[0]['rgb_video'].shape
        collator = IntentCollator(args.extend, image_shape)

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
            pytorch_pickle_regex=args.pytorch_pickle_regex)
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