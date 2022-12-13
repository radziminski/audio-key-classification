import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch import argmax


class AudioClassifier(LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        optimizer,
        scheduler,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = model.num_classes
        self.learning_rate = learning_rate
        self.current_epoch_training_loss = None

    def forward(self, x):
        return self.model(x)

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch)
        preds = torch.argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(
            preds, y, num_classes=self.num_classes, task="multiclass"
        )
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    @staticmethod
    def compute_loss(x, y):
        return torch.nn.CrossEntropyLoss(x, y)

    @staticmethod
    def mirex_score(x, y):
        # TODO https://craffel.github.io/mir_eval/#module-mir_eval.key
        return torch.nn.CrossEntropyLoss(x, y)

    def common_step(self, batch, criterion):
        x, y = batch
        outputs = self(x)
        loss = criterion(outputs, y)
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, self.mirex_score)
        predictions = argmax(outputs, dim=1)
        acc = torchmetrics.functional.accuracy(
            predictions, y, num_classes=self.num_classes, task="multiclass"
        )
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, self.compute_loss)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss}

    def training_epoch_end(self, outs):
        self.current_epoch_training_loss = torch.stack([o["loss"] for o in outs]).mean()

    def validation_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outs):
        if hasattr(self, "current_epoch_training_loss"):
            avg_loss = torch.stack([o["val_loss"] for o in outs]).mean()
            self.logger.experiment.add_scalars(
                "train and vall losses",
                {
                    "train": self.current_epoch_training_loss.item(),
                    "val": avg_loss.item(),
                },
                self.current_epoch,
            )

    def test_step(self, batch, batch_idx):
        loss, acc = self.common_test_valid_step(batch, batch_idx)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return [self.optimizer, self.scheduler]
