import torch
from utils.mirex_loss import mirex_loss_v0, mirex_loss_v1, mirex_loss_v2

MIREX_LOSS_VERSION_MAP = {
    "v0": mirex_loss_v0,
    "v1": mirex_loss_v1,
    "v2": mirex_loss_v2,
}


class MirexLoss(torch.nn.Module):
    def __init__(self, version="v1", device="gpu"):
        super(MirexLoss, self).__init__()
        self.loss = MIREX_LOSS_VERSION_MAP[version]
        self.device = "cuda" if device == "gpu" else device
        self.criterion = (
            torch.nn.CrossEntropyLoss()
            if version == "v1"
            else torch.nn.BCEWithLogitsLoss(reduction="none")
        )

    def forward(self, prediction, true_class):
        return self.loss(prediction, true_class, self.criterion, device=self.device)
