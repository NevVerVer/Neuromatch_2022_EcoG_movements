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
                accumulate_grad_batches={0: 8, 4: 4, 8: 1}):
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
