import  torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric
from ignite.engine import Events

class StatMetric(Metric):

