from .data import FashionMNISTDataModule
from .nn import LitFashionMNIST
import lightning as L

def train():
    # Init DataModule
    dm = FashionMNISTDataModule()
    # Init model from datamodule's attributes
    model = LitFashionMNIST(*dm.dims, dm.num_classes)
    # Init trainer
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
    )
    # Pass the datamodule as arg to trainer.fit to override model hooks :)
    trainer.fit(model, dm)
    '''Надо добавить логгер, calback-и и lr_sheduler'''

