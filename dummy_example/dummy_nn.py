import pytorch_lightning as pl
import torch
from torch.nn import functional as F

class DummyNetwork(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        # You can pass several arguments here using dummy_network.yaml in "conf/network" dir
        
        # Make the neural network here

        # You can also import it from some custom file:
        # from .networks import DummyNet
        # self.model = DummyNet().to(args.dev)

        # Define optimizer and loss (optional)
        # self.loss = ...
        # self.optimizer = ...


    def forward(self, x):
        # Implement the forward step for your neural network from the input x
        
        # If you defined the network before:
        # y = self.forward(x)
        
        # Or in some other way:
        # x = x.view(x.size(0), -1)
        # x = torch.relu(self.l1(x))
        # y = torch.relu(self.l2(x))
        return y

    def training_step(self, batch, batch_idx):
        # You get the input batch and batch index. You must return the batch loss.
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        # You get the input batch. Make your calculation and log what you want.
    
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # Return the optimizer for your model.
        return self.optimizer
