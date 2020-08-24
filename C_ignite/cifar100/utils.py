from contextlib import contextmanager
from datetime import datetime

import os
import sys
import json
import pickle
import re

import six
import numpy as np
import torch

from config import log_keys 

def get_dirname_from_args(args):
    dirname = ''
    for key in sorted(log_keys):
        dirname += '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]

def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')

def prepare_batch(args, batch):
    net_input_key = [*args.use_inputs]
    net_input = {k: batch[k] for k in net_input_key}
    for key, value in net_input.items():
        if torch.is_tensor(value):
            net_input[key] = value.to(args.device).contiguous()

    ans_idx= batch.get('correct_idx', None)
    if torch.is_tensor(ans_idx):
        ans_idx = ans_idx.to(args.device).contiguous()

    return net_input,  ans_index
