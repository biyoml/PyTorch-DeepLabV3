import torch
from utils.constants import VOID_LABEL


class Mean(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, v, n):
        self.sum += v
        self.count += n
        self.result = self.sum / self.count


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.nc = num_classes
        self.mat = torch.zeros([num_classes, num_classes])

    def reset(self):
        self.mat.zero_()

    def update(self, preds, annos):
        preds, annos = preds.cpu(), annos.cpu()
        preds = preds[annos != VOID_LABEL]
        annos = annos[annos != VOID_LABEL]
        indices = preds * self.nc + annos    # x-axis: true label; y-axis: predicted label
        counts = torch.bincount(indices, minlength=self.nc ** 2)
        self.mat += counts.reshape([self.nc, self.nc])

    @property
    def accuracy(self):
        return torch.diag(self.mat).sum() / self.mat.sum()

    @property
    def IoUs(self):
        itersections = torch.diag(self.mat)
        unions = self.mat.sum(0) + self.mat.sum(1) - itersections
        return itersections / unions
