from .data import FashionMNISTDataModule
from .nn import LitFashionMNIST
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 30
CONFIG = dict(
    num_labels=10,
    img_width=28,
    img_height=28,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    activation='ReLU',
    kernels=[5],
    architecture="CNN"
)


def train(path):
    dm = FashionMNISTDataModule(learning_rate=LEARNING_RATE)
    model = LitFashionMNIST(dm.num_classes, dm.learning_rate)
    callbacks = [ModelCheckpoint(dirpath="./checkpoints", every_n_train_steps=1)]
    wandb_logger = WandbLogger(project="fashion_mnist_lit", config=CONFIG)
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        default_root_dir=path,
        logger=wandb_logger,
        callbacks=callbacks
    )
    trainer.fit(model, dm)
