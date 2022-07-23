import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RecurrentAutoencoder(pl.LightningModule): #  nn.Module
    """
    Model:
    SEE: https://github.com/fabiozappo/LSTM-Autoencoder-Time-Series
    SEE: https://github.com/curiousily/Getting-Things-Done-with-Pytorch
    """

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

        # tensorboard writer
        self.writer = SummaryWriter()

        # loss function
        self.f_loss = nn.L1Loss(reduction='sum')

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x

    def training_step(self, batch, batch_idx):
        x = batch

        h = self.encoder(x)
        x_hat = self.decoder(h)

        loss = self.f_loss(x_hat, x)
        self.log('train_loss', loss)
        self.writer.add_scalar('Loss/train', loss, batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch

        h = self.encoder(x)
        x_hat = self.decoder(h)

        loss = self.f_loss(x_hat, x)
        output = dict({
            'test_loss': loss
        })
        self.writer.add_scalar('Loss/test', loss, batch_idx)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)

        return hidden_n.reshape((batch_size, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.shape[0]
        # x = x.repeat(self.seq_len, self.n_features)
        x = x.repeat(self.seq_len, 1)

        x = x.reshape((batch_size, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))

        return self.output_layer(x)


if __name__ == "__main__":
    dataset = torch.randn(50, 10, 2)
    dataset_test = torch.randn(50, 10, 2)

    pl.seed_everything(42)

    # Initialize model and Trainer
    rae = RecurrentAutoencoder(10, 2, 2)
    # test run
    # rae.forward(torch.randn(50, 10, 2))
    trainer = pl.Trainer(max_epochs=10)

    # Perform training
    trainer.fit(rae, DataLoader(dataset, num_workers=4, pin_memory=True))

    # Perform evaluation
    trainer.test(rae, DataLoader(dataset_test, num_workers=4, pin_memory=True))
