import inits
import logging 
import pprint
import datetime
import sys
import random
import os
import torch
import torchvision
import argparse
import yaml

from dataloader.data_factory import get_dataset
from model.model_factory import get_model
from torch.utils import data
from utils.train_utils import get_optimizer_params
from utils.train_utils import LRScheduler, Monitor
from utils import io_utils, eval_utils

def main():
  args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
  #make save_dir
  if not os.path.isdir(args['save_dir']):
      os.makedirs(args['save_dir'])

  #create log file
  log_filename = 'train_records.log'
  log_path = os.path.join(args['save_dir'], log_filename)
  logger = io_utils.get_logger(__name__, log_file=log_path, write_level=logging.INFO, print_level=logging.INFO if args['print_console'] else None, mode='a' if args['resume'] else 'w')
  
  #visda number of classes
  args['num_classes'] = 12

  if args['resume']:
      logger.info('Resume Training')
  else: 
      torch.save(vars(args), os.path.join(args['save_dir'], 'args_dict.pth'))

  num_classes = args['num_classes']
  in_features = num_classses
  num_domains = 2
  num_source_domains = 1
  num_target_domains = 2

  from tensorboardX import SummaryWriter
  tfboard_dir = os.path.join(args['save_dir'], 'tfboard')
  if not os.path.isdir(tfboard_dir):
      os.makedirs(tfboard_dir)
  writer = SummaryWriter(tfboard_dir)

  if args['resume']:
      checkpoints = io_utils.load_latest_checkpoints(args['save_dir'], args, logger)
      start_iter=checkpoints[0]['iteration']+1
  else:
      start_iter=1

  num_domains = len(args['source_datasets'])+len(args['target_datasets'])
  num_source_domains = len(args['source_datasets'])
  num_target_domains = len(args['target_datasets'])

  source_train_datasets = [get_dataset("{}_{}".format(source_name, 'train')) for source_name in args['source_datasets']]
  target_train_datasets = [get_dataset("{}_{}".format(target_name, 'train')) for target_name in args['target_datasets']]

  #dataloader
  source_train_dataloaders = [data.DataLoader(source_train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], drop_last=True, pin_memory=True) for source_train_dataset in source_train_datasets]
  target_train_dataloaders = [data.DataLoader(target_train_dataset, batch_size = args['batch_size'], shuffle=True, num_workers=args['num_workers'], drop_last=True, pin_memory=True) for target_train_dataset in target_train_datasets]
  print("train data loaded")

  #validation dataloader
  target_val_datasets = [get_dataset("{}_{}".format(target_name, 'val')) for target_name in args['target_datasets']]
  target_val_dataloaders = [data.DataLoader(target_val_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'], pin_memory=True) for target_val_dataset  in target_val_datasets]
  print("validation data loaded")

  #model loading
  model = get_model(args['model_name'], args['num_classes'], args['in_features'], num_domains=num_domains, pretrained=True)

  model.train(True)
  if args.resume:
      model.load_state_dict(checkpoints[0]['model'])
  model = model.cuda(args['gpu'])

  optimizer = optim.SGD(params, momentum=0.9, nesterov=True)

  logger.info('Train Starts')

  monitor = Monitor()

  global best_accuracy
  best_accuracy = 0.0

  for i_iter in range(epochs):
      optimizer.zero_grad()


if __name__=='__main__':
  main()
