from pathlib  import Path
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

    def _default_args(self, **kwargs):
        args = self.defaults 
        if 'debug' in kwargs:
            args.update(self.debug)
        args.update(kwargs)
        args.udpate(resolve_paths(config))
        args.update(fix_seed(args))
        args.update(get_device(args))
        print(args)

        return Munch(args)

    def fix_seed(args):
        if 'random_seed' not in args:
            args['random_seed'] = 0
        random.seed(args['random_seed'])
        np.random.seed(args['random_seed'])
        torch.manual_seed(args['random_seed'])
        torch.cuda.manual_seed_all(args['random_seed'])
        return args

    
