import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.utils.mirex import (
    mirex_score,
    get_maps,
    get_perfect_fifths,
    get_relative,
    get_parallel,
)

class_to_idx = {
    "A-maj": 0,
    "A-min": 1,
    "Ab-maj": 2,
    "Ab-min": 3,
    "B-maj": 4,
    "B-min": 5,
    "Bb-maj": 6,
    "Bb-min": 7,
    "C-maj": 8,
    "C-min": 9,
    "D-maj": 10,
    "D-min": 11,
    "Db-maj": 12,
    "Db-min": 13,
    "E-maj": 14,
    "E-min": 15,
    "Eb-maj": 16,
    "Eb-min": 17,
    "F-maj": 18,
    "F-min": 19,
    "G-maj": 20,
    "G-min": 21,
    "Gb-maj": 22,
    "Gb-min": 23,
}
idx_to_class = {v: k for k, v in class_to_idx.items()}


class KeyClassifier(LightningModule):
    def __init__(
        self,
        model,
        learning_rate,
        optimizer,
        scheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_start(self):
        self.val_acc_best.reset()

    def step(self, batch):
        keys_idx_map, keys_idx_map_inv = get_maps()
        x, y = batch
        mirex_y = torch.zeros(len(y), 24).to("cuda")
        for index, y_true_t in enumerate(y):
            y_true = int(y_true_t)
            mirex_y[index][y_true] = 0.5

            true_key = keys_idx_map[y_true]
            first, second = get_perfect_fifths(true_key)
            first, second = keys_idx_map_inv[first], keys_idx_map_inv[second]

            relative = keys_idx_map_inv[get_relative(true_key)]
            parallel = keys_idx_map_inv[get_parallel(true_key)]

            mirex_y[index][first] = 0.175
            mirex_y[index][second] = 0.175
            mirex_y[index][relative] = 0.1
            mirex_y[index][parallel] = 0.05

        logits = self.forward(x)
        loss = self.criterion(logits, mirex_y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y

    def _test_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        loss = mirex_score(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        loss, preds, targets = self._test_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss-mirex",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

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
