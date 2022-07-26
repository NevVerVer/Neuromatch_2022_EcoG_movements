import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LinearAutoencoder(pl.LightningModule):
    def __init__(self, n_input, n_embedding=16):
        super(LinearAutoencoder, self).__init__()

        # parameters
        self.lr = 1e-2

        # layers
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 8*n_embedding),
            nn.ReLU(True),
            nn.Linear(8*n_embedding, 4*n_embedding),
            nn.ReLU(True),
            nn.Linear(4*n_embedding, n_embedding))
        self.decoder = nn.Sequential(
            nn.Linear(n_embedding, 4*n_embedding),
            nn.ReLU(True),
            nn.Linear(4*n_embedding, 8*n_embedding),
            nn.ReLU(True),
            nn.Linear(8*n_embedding, n_input),
            nn.Tanh())

        # loss
        self.custom_loss = nn.L1Loss(reduction='sum')

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat = self.forward(x)

        loss = self.custom_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        x_hat = self.forward(x)

        loss = self.custom_loss(x_hat, x)
        self.log('validation_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch

        x_hat = self.forward(x)

        loss = self.custom_loss(x_hat, x)
        output = dict({
            'test_loss': loss
        })
        self.log('test_loss', loss)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer),
                "monitor": "validation_loss",
                "frequency": 1
            },
        }