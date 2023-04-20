import torch
from pytorch_lightning.core.datamodule import LightningDataModule
import os


class DummyDataModule(LightningDataModule):
    def __init__(self, args):
        # Load your dataset and create the "train_dataloader" and "test_dataloader".
        # You can use a PyTorch Dataset class (see DummyDataset example in dummy_data_utils.py)
        # You can pass several args here using the dummy_dataset.yaml in "conf/db" dir

        # train_d = DummyDataset(...)
        # val_d = DummyDataset(...)
        # test_d = DummyDataset(...)
        # self.train_dl = torch.utils.data.DataLoader(train_d)
        # self.val_dl = torch.utils.data.DataLoader(val_d)
        # self.test_dl = torch.utils.data.DataLoader(test_d)
        
    def prepare_data(self):
        0
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

    def setup(self):
        0
        
    def train_dataloader(self):
        # Return the train dataloader
        return self.train_dl

    def val_dataloader(self):
        # Return the val dataloader 
        return self.val_dl

    def test_dataloader(self):
        # Return the test dataloader 
        return self.test_dl

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        0