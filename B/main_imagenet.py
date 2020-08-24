import os
import torch
import torchvision
import argparse
import yaml
from torch.utils import data

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.imagenet.mlp import MLP

def main():
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  args = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
   
  ############Data Loading#########################################
  traindir = os.path.join(args['root'], 'images', 'train')
  valdir = os.path.join(args['root'], 'images', 'val')
  normalize = transforms.Normalize(mean=[0.485,  0.456, 0.406],
                                   std=[0.229,0.224,0.225])

  trainloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir, transforms.Compose([
      transforms.RandomSizedCrop(224), 
      transforms.RandomHorizontalFlip(), 
      transforms.ToTensor(), 
      normalize,
    ])),
    batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)
  
  valloader = torch.utils.data.DataLoader(
    datsets.ImageFolder(valdir, transforms.Compose([
      transforms.Scale(256),
      transforms.CenterCrop(224), 
      transforms.ToTensor(),
      normalize,
    ])),
    batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)
  print('preparing dataset imagenet completed')

  ####################creating  model###############################3
  model = MLP(args['model_name'], args['pretrained'], 1000)
  criterion = nn.CrossEntropyLoss()
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
