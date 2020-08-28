# My question is how does iters turn to one batch?

import ignite.engine.engine import Engine, State, Events

from ckpt import get_model_ckpt, save_ckpt
from loss import get_loss
from optimizer import get_optimizer
from logger import get_logger, log_results, log_results_cmd

from utils import prepare_batch
from metric import get_metrics

from evaluate import get_evaluator, evaluate_once
from metric.stat_metric import StatMetric
import numpy as np


