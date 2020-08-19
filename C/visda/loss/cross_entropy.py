import math
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from metric.stat_metric import StatMetric

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, eps=0, padding_idx=0):
        super(CrossEntropyLoss, self).__init__(ignore_index=padding_idx)
        self.eps = eps
        self.padding_idx = padding_idx


