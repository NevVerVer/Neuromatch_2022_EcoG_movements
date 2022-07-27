import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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


def plot_reach(ax, data, event, plot_ticks_and_labels=True, plot_line=True):
    x = data[event, :, 0]
    y = data[event, :, 1]
    if plot_line:
        ax.plot(x, y, '-', alpha=0.5)
    ax.scatter(x, y, c=np.arange(len(x)), alpha=0.5)
    ax.set_aspect('equal')
    if plot_ticks_and_labels:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(event)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
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
        plot_reach(ax[0, i], data, idx, plot_ticks_and_labels=False)
        with torch.no_grad():
            # Get reconstructed movements from autoencoder
            recon = model(data_[idx:idx+1, :])[0]
            if plot_latent:
                z, mu, log_var = model.encode(data_[idx:idx+1, :])
                plot_latent_space(ax[2, i], z)

        plot_reach(ax[1, i], torch.swapaxes(
            recon.reshape((2, 75)), 1, 0).unsqueeze(0), 0,
                   plot_ticks_and_labels=False)
        ax[0, i].set_title(idx)

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
        inx = np.concatenate([z['ind'][:n_ex], z['ind'][-n_ex:]])

        for ii in range(n_ex * 2):
            plot_reach(ax[i, ii], data, inx[ii], plot_ticks_and_labels=False)
        ax[i, 0].set_ylabel(z_name)

    plt.show()


def plot_data_in_latent_space(model, data, n_clusters=3):
    data_ = torch.tensor(data, device='cpu', dtype=torch.float)
    data_ = torch.swapaxes(data_, 2, 1).view(data_.size(0), -1)
    plt.figure(figsize=(7, 7))

    with torch.no_grad():
        rec = model(data_)
        z, mu, log_var = model.encode(data_)

    if z.shape[1] > 2:
        print('Running t-SNE')
        z_orig = z.clone()
        z = TSNE(n_components=2).fit_transform(z.numpy())

    if n_clusters > 0:
        kmeans = KMeans(n_clusters=n_clusters)
        labels_ = kmeans.fit_predict(z)
        plt.scatter(z[:, 0], z[:, 1], c=labels_, alpha=0.5)
        plt.colorbar()
        plt.show()
        return labels_, z_orig
    else:
        plt.scatter(z[:, 0], z[:, 1], alpha=0.5)
        plt.show()
        return None


def plot_examples_from_class(labels, data, n_ex=5):
    clusters = np.unique(labels)
    fig, ax = plt.subplots(1, len(clusters),figsize=(20, 6))

    for i, l in enumerate(clusters):
        for ex in np.where(labels == l)[0][:n_ex]:
            plot_reach(ax[i], data, ex, plot_ticks_and_labels=False,
                       plot_line=False)
        ax[i].set_title(f'Cluster {l}')
    plt.show()
