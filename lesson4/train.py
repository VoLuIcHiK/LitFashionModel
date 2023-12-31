from .data import FashionMNISTDataModule
from .nn import LitFashionMNIST
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

LEARNING_RATE = 0.0001
BATCH_SIZE = 64


def train(path, epochs):
    config = dict(
        num_labels=10,
        img_width=28,
        img_height=28,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        activation='ReLU',
        kernels=[5],
        architecture="CNN"
    )
    dm = FashionMNISTDataModule()
    model = LitFashionMNIST(num_classes=dm.num_classes,
                            len_train=55000,
                            batch_size=BATCH_SIZE,
                            epochs=epochs)
    callbacks = [ModelCheckpoint(dirpath="./checkpoints",
                                 every_n_train_steps=1)]
    wandb_logger = WandbLogger(project="fashion_mnist_lit",
                               config=config)
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=path,
        logger=wandb_logger,
        callbacks=callbacks
    )
    trainer.fit(model, dm)
