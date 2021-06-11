
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class ZabitModule(pl.LightningModule):
    def __init__(self, num_classes=4, lr=3e-4):
        super().__init__()

        self.norm_layer = nn.LayerNorm
        self.activation = nn.ReLU

        self.backbone = nn.Sequential( # CBR-Tiny arXiv:1902.07208
            nn.Conv2d(3, 64, 5),
            self.norm_layer([124, 124]),
            self.activation(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 256, 5),
            self.norm_layer([57, 57]),
            self.activation(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 512, 5),
            self.norm_layer([24, 24]),
            self.activation(),
            nn.MaxPool2d(3, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.seq_model = nn.LSTM(512, 64, 2, dropout=0.3, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(2 * 64, num_classes)
        )
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, input_tensor):
        b, t, f, h, w = input_tensor.shape

        hidden = self.backbone(input_tensor.view(b * t, f, h, w))
        output_seq, _ = self.seq_model(hidden.view(b, t, -1))

        return self.classifier(output_seq.mean(axis=1))

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=0)

        return [optimizer], [scheduler]
