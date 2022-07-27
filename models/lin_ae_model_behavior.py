import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class LinearAutoencoder(pl.LightningModule):
    def __init__(self, n_input, n_hidden=16):
        super(LinearAutoencoder, self).__init__()

        # parameters
        self.lr = 1e-2

        # layers
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 8 * n_hidden),
            nn.ReLU(True),
            nn.Linear(8 * n_hidden, 4 * n_hidden),
            nn.ReLU(True),
            nn.Linear(4 * n_hidden, n_hidden))
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, 4 * n_hidden),
            nn.ReLU(True),
            nn.Linear(4 * n_hidden, 8 * n_hidden),
            nn.ReLU(True),
            nn.Linear(8 * n_hidden, n_input),
            nn.Tanh())

        # loss

    def loss_function(self, recons, input):
        loss = F.mse_loss(recons, input)
        return loss

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x

    def training_step(self, batch, batch_idx):
        x = batch

        x_hat = self.forward(x)

        loss = self.loss_function(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        x_hat = self.forward(x)

        loss = self.loss_function(x_hat, x)
        self.log('validation_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch

        x_hat = self.forward(x)

        loss = self.loss_function(x_hat, x)
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


class LinearVariationalAutoencoder(LinearAutoencoder):
    def __init__(self, n_input, n_hidden=16, n_latent=5, beta: int = 4,
                 gamma: float = 1000.,  max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5, loss_type: str = 'B'):
        super(LinearAutoencoder, self).__init__(n_input, n_hidden)

        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_var = nn.Linear(n_hidden, n_latent)

    def forward(self, input):
        result = self.encoder(input)

        mean = self.mean_layer(result)
        log_var = self.std_layer(result)

        z = self.reparameterize(mean, log_var)

        return [self.decoder(z), input, mean, log_var]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self, *args, **kwargs):
        # SEE: https://github.com/AntixK/PyTorch-VAE
        self.num_iter += 1
        recons = args[0]
        inp = args[1]
        mu = args[2]
        log_var = args[3]
        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']

        recons_loss = F.mse_loss(recons, inp)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1),
            dim=0)

        # https://openreview.net/forum?id=Sy2fzU9gl
        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss

        # https://arxiv.org/pdf/1804.03599.pdf
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(inp.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0,
                            self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss
