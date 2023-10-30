import os
import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ExponentialLR
PATH_DATASETS = './lesson4'
BATCH_SIZE = 64


class LitFashionMNIST(L.LightningModule):
    def __init__(self, num_classes, learning_rate=2e-4):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        return lr_scheduler
