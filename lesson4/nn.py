import os
import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ExponentialLR, OneCycleLR
PATH_DATASETS = './lesson4'
BATCH_SIZE = 64


class LitFashionMNIST(L.LightningModule):
    def __init__(self, num_classes, len_train, batch_size, epochs):
        super().__init__()
        self.num_classes = num_classes
        self.len_train = len_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = nn.Sequential(
            #(channels * width * height)
            #input (1 * 28 * 28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            # output (16 * 24 * 24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # input (16 * 12 * 12)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            # output (32 * 8 * 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output (32 * 4 * 4)
            nn.Flatten(), #преобразовали в тензор размера (4 * 4, 1)
            nn.Linear(in_features=4 * 4 * 32, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        n = 10 #выведем в логирование 10 изображений
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        images = [img for img in x[:n]]
        captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], preds[:n])]
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        #self.logger.log_image(self, key='sample_images', images=images, caption=captions)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        #lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=0.01,
            steps_per_epoch=2 * self.len_train,
            epochs=self.epochs,
            three_phase=True
        )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        #return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        #steps_per_epoch=self.len_train//(self.batch_size * 8)
