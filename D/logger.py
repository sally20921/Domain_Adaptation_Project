import os
import logging 
from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf
from utils import get_dirname_from_args, get_now

class Logger:
    def __init__(self, args):
        self.log_cmd = args.log_cmd
        log_name = get_dirname_from_args(args)
        log_name+= '_{}'.format(get_now())
        self.log_path = args.log_path / log_name

