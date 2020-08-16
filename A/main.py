import os
import torch
import torchvision
import argparse
import yaml
#from utils import post_config_hook

from dataloader.data_factory import get_dataset
from torch.utils import data
#@ex.automain
def main():
  #args = argparse.Namespace(**_run.config)
  #args = post_config_hook(args, _run)
  #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #args.n_gpu = torch.cuda.device_count()
  args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

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
if __name__=='__main__':
  main()
