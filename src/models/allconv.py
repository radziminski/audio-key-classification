import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import argmax


class AllConv(LightningModule):
    def __init__(self, num_feature_maps, dropout_rate, learning_rate, input_channels=1, num_classes=24):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.current_epoch_training_loss = None

        nf = num_feature_maps
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=nf, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=nf, out_channels=2 * nf, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=2 * nf, out_channels=2 * nf, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=2 * nf, out_channels=4 * nf, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=4 * nf, out_channels=4 * nf, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=4 * nf, out_channels=8 * nf, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=8 * nf, out_channels=8 * nf, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=8 * nf, out_channels=num_classes, kernel_size=1, padding=1)
        self.global_pool_avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv7(x))
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv8(x))
        x = F.dropout(x, p=self.dropout_rate)

        x = F.elu(self.conv9(x))
        x = self.global_pool_avg(x).squeeze(3).squeeze(2)
        return x

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch)
        preds = torch.argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(preds, y, num_classes=self.num_classes, task='multiclass')
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    @staticmethod
    def compute_loss(x, y):
        return F.cross_entropy(x, y)

    @staticmethod
    def mirex_score(x, y):
        # TODO https://craffel.github.io/mir_eval/#module-mir_eval.key
        return F.cross_entropy(x, y)

    def common_step(self, batch, criterion):
        x, y = batch
        outputs = self(x)
        loss = criterion(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, self.mirex_score)
        predictions = argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(predictions, y, num_classes=self.num_classes, task='multiclass')
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, self.compute_loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, outs):
        self.current_epoch_training_loss = torch.stack([o["loss"] for o in outs]).mean()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outs):
        if hasattr(self, 'current_epoch_training_loss'):
            avg_loss = torch.stack([o["val_loss"] for o in outs]).mean()
            self.logger.experiment.add_scalars('train and vall losses',
                                               {'train': self.current_epoch_training_loss.item(),
                                                'val': avg_loss.item()}, self.current_epoch)

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return [optimizer]