import torch
from torchmetrics import Metric
from src.utils import mirex


class MirexMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mirex", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.mirex += mirex.mirex_score(target, preds).sum()
        self.total += target.numel()

    def compute(self):
        return self.mirex.float() / self.total
