import pytorch_lightning as pl
import torch
from torch.nn import functional as F

class LitClassifier(pl.LightningModule):
    """
    Prirejeno po:
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/simple_image_classifier.py
    
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        learning_rate: float = 0.0001,
        input_shape: tuple = (28,28)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(self.hparams.input_shape[0] * self.hparams.input_shape[1], self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
