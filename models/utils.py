import numpy as np
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
