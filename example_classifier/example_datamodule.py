import torch
from pytorch_lightning.core.datamodule import LightningDataModule
import os


class MNISTDataModule(LightningDataModule):
    def __init__(self):
        # Majhen wrapper okoli MNISTDataModule of pytorch-lightning
        from pl_examples.basic_examples.mnist_datamodule import MNISTDataModule
        mnist_module = MNISTDataModule()
        mnist_module.setup()
        train_d, val_d = mnist_module.dataset_train,mnist_module.dataset_val

        self.train_dl = torch.utils.data.DataLoader(train_d)
        self.val_dl = torch.utils.data.DataLoader(val_d)

        
    def prepare_data(self):
        0
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
    def setup(self):
        0
        
    def train_dataloader(self):
        
        return self.train_dl
    def val_dataloader(self):
        return self.val_dl
    def test_dataloader(self):
        
        return self.val_dl
    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        0