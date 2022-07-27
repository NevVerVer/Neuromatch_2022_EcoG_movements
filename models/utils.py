import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


def train_model(model,
                X_train,
                X_val,
                X_test,
                n_epochs=20,
                batch_size=20,
                add_to_log_name=None,
                accumulate_grad_batches=None):
    if accumulate_grad_batches is None:
        accumulate_grad_batches = {0: 8, 4: 4, 8: 1}
    if add_to_log_name is None:
        add_to_log_name = ['n_latent', 'lr']

    # setup logger name
    name = model.__class__.__name__

    for par in add_to_log_name:
        if hasattr(model, par):
            name += f'_{par}={getattr(model, par)}'

    # setup logger
    logger = TensorBoardLogger("tb_logs", name=name)

    # setup torch lightning trainer
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator='mps',
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches
    )

    # Perform training
    trainer.fit(
        model,
        DataLoader(X_train, batch_size=batch_size, shuffle=True),
        DataLoader(X_val, batch_size=batch_size, shuffle=False)
    )

    # Perform evaluation
    trainer.test(model, DataLoader(X_test, shuffle=False))


def plot_reach(ax, data, event):
    x = data[event, :, 0]
    y = data[event, :, 1]
    ax.plot(x, y, '-', alpha=0.5)
    ax.scatter(x, y, c=np.arange(len(x)), alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(event)
    # set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])


def plot_reconstruction_examples(model, data, n_examples=10):
    n_plot = n_examples
    fig, ax = plt.subplots(2, n_plot, figsize=(20, 6))
    data_ = torch.tensor(data, device='cpu', dtype=torch.float)
    data_ = torch.swapaxes(data_, 2, 1).view(data_.size(0), -1)

    for i in range(n_plot):
        idx = torch.randint(len(data), size=())

        plot_reach(ax[0, i], data, idx)

        with torch.no_grad():
            # Get reconstructed movements from autoencoder
            recon = model(data_[idx:idx+1, :])[0]

        plot_reach(ax[1, i], torch.swapaxes(
            recon.reshape((2, 75)), 1, 0).unsqueeze(0), 0)
        ax[0, i].set_title(idx)
        ax[1, i].set_title('')
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[0, i].set_xlabel('')
        ax[0, i].set_ylabel('')
        ax[1, i].set_xlabel('')
        ax[1, i].set_ylabel('')

        if i == 0:
            ax[0, i].set_ylabel('Original\nMovements')
            ax[1, i].set_ylabel(f'Reconstructed\nMovements')

    plt.show()
