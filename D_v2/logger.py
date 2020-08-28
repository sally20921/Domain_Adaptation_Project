import logging
import os 

from tqdm import tqdm 
from tensorboardX import SummaryWriter
from tensorboard import default, program
import tensorflow as tf

from utils import get_dirname_from_args, get_now

class Logger:
    def __init__(self, args):
        self.log_cmd = args.log_cmd
        log_name  = get_dirname_from_args(args)
        log_name += '_{}'.format(get_now())
        self.log_path = args.log_path / log_name
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.tfboard = SummaryWriter(self.log_path)

        self.url = run_tensorboard(self.log_path)
        print("Running Tensorboard at {}".format(self.url))

    def __call__(self, name, val, n_iter):
        self.tfboard.add_scalar(name, val, n_iter)
        if self.log_cmd:
            tqdm.write('{}:({},{})'.format(n_iter, name, val))

def run_tensorboard(log_path):
    log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log = logging.getLogger('tensorflow').setLevel(logging.ERROR)

    
def log_results(logger, name, state, step):
# log results in log file

def log_results_cmd(name, state, step):
# log results in command line

def get_logger(args):
    return Logger(args)
