import os
from pathlib import Path

from torch import optim
from ignite.metrics.metric import Metric

from inflection import underscore

metric_dict = {}


