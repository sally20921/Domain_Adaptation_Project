import os
import random
import shutil
import time
import warnings

import torchvision
import argparse
import yaml
from torch.utils import data

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.mlp import MLP

from tensorboardX import SummaryWriter

def main():
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
  
  ############Data Loading#########################################
  dataloader =  datasets.CIFAR100
  num_classes = 100

  transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
  transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
 
  trainset = dataloader(root=args['root'], train=True, download=True, transform = transform_train)
  trainloader = data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])

  testset =  dataloader(root=args['root'], train=False, download=False, transform=transform_test)
  testloader = data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])
  print('preparing dataset cifar completed')
  ####################creating  model###############################3
  model = MLP(args['model_name'], num_classes)
  model.cuda()

  criterion = nn.CrossEntropyLoss().cuda()
  optimizer =  optim.SGD(model.parameters(), lr=args['learning_rate'],momentum=args['momentum'], weight_decay=args['weight_decay'])

  for epoch in range(args['epochs']):
    print('epoch: ', epoch+1)
    train(trainloader, model, criterion, optimizer, epoch)
    test(testloader, model, criterion, epoch)

def train(trainloader, model, criterion, optimizer, epoch):
  model.train()
  
  for batch_idx, (inputs, targets) in enumerate(trainloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test(testloader, model, criterion, epoch):
  model.eval()
  outputs = model(inputs)
  loss = criterion(outputs, targets)
  

if __name__ == '__main__':
  main()
