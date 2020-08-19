import os
import inspect
from pathlib import Path

from torch.nn.modules.loss import _Loss

from inflection import underscore

loss_dict = {}


