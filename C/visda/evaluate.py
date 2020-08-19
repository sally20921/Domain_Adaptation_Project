import  torch
from ignite.engine.engine import Engine,  State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics

def get_evaluator(args, model, loss_fn, metrics={}):
    from termcolor import colored

    sample_count = 0


