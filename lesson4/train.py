from .data import FashionMNISTDataModule
from .nn import LitFashionMNIST
import lightning as L
import wandb
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 10
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

def train(PATH):
    run = wandb.init(project="fashion_mnist_lit", reinit=True, config=CONFIG)
    config = run.config
    dm = FashionMNISTDataModule(learning_rate=LEARNING_RATE)
    config.learning_rate = LEARNING_RATE
    model = LitFashionMNIST(dm.num_classes, dm.learning_rate)
    #добавить в wandb "loss": loss
    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        default_root_dir=PATH,
        logger=run
    )
    trainer.fit(model, dm)
    checkpoint_callback = L.ModelCheckpoint(dirpath=PATH, save_top_k=2, monitor="val_loss")
    run.finish()
    #checkpoint_callback.best_model_path
    '''Надо добавить логгер, calback-и и lr_sheduler'''

