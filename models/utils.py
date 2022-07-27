import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


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


def plot_latent_space(ax, z):
    ax.bar(np.arange(z.shape[1]), z[0, :])
    ax.set_xlabel('Latent Space (z)')
    ax.set_xticks(np.arange(z.shape[1]))
    ax.set_ylim([-2, 2])


def plot_reconstruction_examples(model, data, n_examples=10, plot_latent=False):
    n_plot = n_examples
    fig, ax = plt.subplots(3 if plot_latent else 2, n_plot, figsize=(20, 6))
    data_ = torch.tensor(data, device='cpu', dtype=torch.float)
    data_ = torch.swapaxes(data_, 2, 1).view(data_.size(0), -1)

    for i in range(n_plot):
        idx = torch.randint(len(data), size=())

        plot_reach(ax[0, i], data, idx)

        with torch.no_grad():
            # Get reconstructed movements from autoencoder
            recon = model(data_[idx:idx+1, :])[0]
            if plot_latent:
                z, mu, log_var = model.encode(data_[idx:idx+1, :])
                plot_latent_space(ax[2, i], z)

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


def plot_examples_based_on_latent_space(model, data, n_ex=5):
    data_ = torch.tensor(data, device='cpu', dtype=torch.float)
    data_ = torch.swapaxes(data_, 2, 1).view(data_.size(0), -1)
    with torch.no_grad():
        rec = model(data_)
        z, mu, log_var = model.encode(data_)

    z = z.numpy()
    z_names = [f'z{z + 1}' for z in np.arange(z.shape[1])]

    fig, ax = plt.subplots(z.shape[1], n_ex * 2, figsize=(20, 7))

    z = np.core.records.fromarrays(
        [np.arange(z.shape[0])] + [zi for zi in z.T],
        names='ind,'+','.join(z_names))

    for i, z_name in enumerate(z_names):
        z.sort(order=z_name)
        # zi = z['z1'][:n_ex * 2]
        # zi = np.concatenate([z['z1'][:n_ex], z['z1'][-n_ex:]])
        inx = np.concatenate([z['ind'][:n_ex], z['ind'][-n_ex:]])

        for ii in range(n_ex * 2):
            plot_reach(ax[i, ii], data, inx[ii])
            ax[i, ii].set_xticks([])
            ax[i, ii].set_yticks([])
            ax[i, ii].set_xlabel('')
            ax[i, ii].set_ylabel('')
        ax[i, 0].set_ylabel(z_name)

    plt.show()


def plot_data_in_latent_space(model, data, n_clusters=3):
    data_ = torch.tensor(data, device='cpu', dtype=torch.float)
    data_ = torch.swapaxes(data_, 2, 1).view(data_.size(0), -1)


    with torch.no_grad():
        rec = model(data_)
        z, mu, log_var = model.encode(data_)

    if n_clusters > 0:
        kmeans = KMeans(n_clusters=n_clusters)
        labels_ = kmeans.fit_predict(z)
        plt.scatter(z[:, 0], z[:, 1], c=labels_)
        plt.colorbar()
    else:
        plt.scatter(z[:, 0], z[:, 1])
    plt.show()