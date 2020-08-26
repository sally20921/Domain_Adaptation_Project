import json

import torch
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from ckpt import get_model_ckpt
from model import get_model
from loss import get_loss
from logger import log_results_cmd

from utils import prepare_batch
from metric import get_metrics

def get_evaluator(args, model, loss_fn):
    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            net_inputs, _ = prepare_batch(args, batch)
            y_pred = model(**net_inputs)
            y_pred = y_pred.argmax(dim=-1)


    engine = Engine(_inference)
    
    return engine


