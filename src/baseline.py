#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class BaselineModule(pl.LightningModule):
    def __init__(self, input_size, num_classes=4, lr=3e-4):
        super().__init__()

        self.backbone = nn.Sequential( # CBR-Tiny arXiv:1902.07208
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        hidden_size = self._get_hidden_size(input_size)

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def _get_hidden_size(self, input_size):
        self.backbone(torch.randn(1, 3, input_size, input_size))

    def forward(self, input_tensor):
        hidden = self.backbone(input_tensor)
        return self.classifier(hidden.squeeze())

    def training_step(self, batch, batch_idx):
        input_tensor, target = batch

        logits = self(input_tensor)
        loss = F.cross_entropy(logits, target)

        self.train_acc(F.softmax(logits, 1), target)
        self.log('train_acc', self.train_acc, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor, target = batch

        logits = self(input_tensor)
        loss = F.cross_entropy(logits, target)

        self.val_acc(F.softmax(logits, 1), target)
        self.log('val_acc', self.val_acc, on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
