import os
import logging
import argparse
import shutil

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

from logger import setup_logger
from dataloader.load_dataset import Datasets
from model.simclr import SimCLR
from loss.nt_xent import NTXent

def main():
    args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    args['base']['logger_name'] = 'simclr'

    if not os.path.exists(args['base']['output_dir_path']):
        os.makedirs(args['base']['output_dir_path'])

    if not  os.path.exists(args['base']['log_dir_path']):
        os.makedirs(args['base']['log_dir_path'])

    logger = setup_logger(args['base']['logger_name'], args['base']['log_file_path'])
    logger.info('using config: {}'.format(args))

    writer =  SummaryWriter(log_dir=args['base']['log_dir_path'])


if __name__ == '__main__':
    main()
