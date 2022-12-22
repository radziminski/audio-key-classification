import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .components.mirex_metric import MirexMetric


class KeyClassifier(LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        optimizer,
        scheduler,
        criterion,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "criterion"])
        self.model = model
        self.criterion = criterion
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        self.train_mirex = MirexMetric()
        self.val_mirex = MirexMetric()
        self.test_mirex = MirexMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        self.val_acc_best.reset()

    def step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def _test_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_mirex(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mirex", self.train_mirex, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_mirex(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/mirex", self.val_mirex, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        loss, preds, targets = self._test_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_mirex(preds, targets)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/mirex",
            self.test_mirex,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
