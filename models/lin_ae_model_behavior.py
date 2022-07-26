import torch
from torch import nn
import torch.nn.functional as F
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
        self.custom_loss = nn.MSELoss()  # nn.L1Loss(reduction='sum')

    # def custom_loss(self, ae_input, ae_output):
    #     inp = ae_input.view(ae_input.shape[0], 2, -1)
    #     out = ae_output.view(ae_output.shape[0], 2, -1)
    #
    #     # l1 loss
    #     l1_loss = F.l1_loss(inp, out, reduction='sum')
    #
    #     # maximize average cosine similarity
    #     # cos_sim = 0
    #     # for (b1, b2) in zip(ae_input, ae_output):
    #     #     cos_sim += 1 - F.cosine_similarity(b1, b2).mean()
    #
    #     # loss end
    #     l1_loss_end = F.l1_loss(
    #         inp, out, reduction='none')[:, -1, :].sum()
    #
    #     # loss start
    #     l1_loss_start = F.l1_loss(
    #         inp, out, reduction='none')[:, 0, :].sum()
    #
    #     # additional penalty for the max l1
    #     # l1_loss_max = F.l1_loss(
    #     #     ae_input, ae_output, reduction='none').max()
    #
    #     # cos_sim +  # l1_loss # + l1_loss_max  # cos_sim
    #     alpha = 0.8
    #     loss = l1_loss * alpha + (l1_loss_end + l1_loss_start) * (1 - alpha)
    #     return loss

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