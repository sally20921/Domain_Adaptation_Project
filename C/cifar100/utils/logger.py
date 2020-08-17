import os
import logging 

from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboard import default, program
import tensorflow as tf 

class Logger:
  def __init__(self, args):
    
