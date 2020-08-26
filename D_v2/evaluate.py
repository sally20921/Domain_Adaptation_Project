import torch
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_result_cmd

from utils import prepare_batch
from metric import  get_metrics

def get_evaluator(args, model, loss_fn, metrics={}):
    # for coloring terminal output 
    from termcolor import colored

    sample_count = 0

    def _inference(evaluator, batch):
        nonlocal sample_count

        model.eval()
        with torch.no_grad():
            net_inputs, target = prepare_batch(args, batch)
            y_pred = model(**net_inputs)
            batch_size = y_pred.shape[0]
            loss, stats = loss_fn(y_pred, target)

        return loss.item(), stats, batch_size, y_pred, target

    engine = Engine(_inference)





