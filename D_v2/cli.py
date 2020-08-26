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
from infer import  infer

class Cli:
    def __init__(self):
        self.defaults = config
        self.debug = debug_options

    def _default_args(self, **kwargs):
        args = self.defaults 
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args.update(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))
        print(args)

        return Munch(args)

    def check_dataloader(self, **kwargs):

