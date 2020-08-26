import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engine import Events

class StatMetric(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(StatMetric, self).__init__(output_transform)

        self.log_iter = True

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        average_loss = output[0]
        N = output[1]

        self._sum += average_loss * N
        self._num_examples += N

    def compute(self):


