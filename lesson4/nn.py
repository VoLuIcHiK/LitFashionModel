import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
PATH_DATASETS = './lesson4'
BATCH_SIZE = 64


class LitFashionMNIST(L.LightningModule):
    def __init__(self, channels, width, height, num_classes, learning_rate=2e-4):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = nn.Sequantial(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Linear(in_features=256, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer