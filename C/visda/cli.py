from pathlib import Path
import random

from fire import Fire
from munch import Munch

import torch
import numpy as np

from config import config, debug_options 
from dataloader.load_dataset import get_iterator
from utils import wait_for_key, suppress_stdout
from train import train
from evaluate import evaluate
from infer import infer

class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options 

    def train(self, **kwargs):
        args = self._default_args(**kwargs)
        train(args)
        wait_for_key()

    def evaluate(self, **kwargs):
        args = self._default_args(**kwargs)
        evaluate(args)
        wait_for_key()

    def infer(self, **kwargs):
        args = self._default_args(**kwargs)
        infer(args)

def resolve_paths(config):
    paths = [k for  k in config.keys() if k.endswith('_path')]
    res = {}
    root = Path('../').resolve()
    for path in paths:
        res[path] = root / config[path]
    return res


