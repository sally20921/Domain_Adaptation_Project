'''
factory method for easily getting dataset by name and type
'''
from dataloader.load_dataset import OFFICE, OFFICEHOME, VISDA

import utils.custom_transforms as custom_transforms
from torchvision import transforms 
from numpy import array 

__sets = {}

OFFICE_DIR = './datasets/'
OFFICEHOME_DIR = './datasets/Office-Home/'
VISDA_DIR = './datasets/VisDA/'
SPLITS = ['train', 'test', 'val']

#for Office DA
for split in SPLITS:
  # for resnet
  if split == 'train':
    transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                           std=[0.229,0.224, 0.225])
    ])
  else:
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224), 
      transforms.ToTensor(), 
      transforms.Normalize(mean=[0.485,0.456,0.406],
                           std=[0.229,0.224,0.225])
    ])
  for domain in ['amazon', 'webcam', 'dslr']:
    name = '{domain}_{split}'.format(domain=domain, split=split)
    __sets[name] = (lambda domain=domain, transform=transform: OFFICE(OFFICE_DIR, domain=domain, transform=transform))

#for visda

