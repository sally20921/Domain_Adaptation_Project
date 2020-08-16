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
for split in SPLITS:
  if split == 'train':
    transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
  else:
    transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(227),
                            custom_transforms.PILToNumpy(),
                            custom_transforms.RGBtoBGR(),
                            custom_transforms.SubtractMean(
                                mean=array([104.0069879317889, 116.66876761696767, 122.6789143406786])),
                            custom_transforms.NumpyToTensor()
                        ])
  for domain in ['train', 'validation', 'test']:
    name = '{domain}_{split}'.format(domain=domain, split=split)

    __sets[name] = (
      lambda domain=domain, transform=transform: VISDA(VISDA_DIR, domain=domain, transform=transform))

#Office Home
for split in SPLITS:
  # for resnet
  if split == 'train':
    ransform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
  else:
    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
  for domain in ['Art', 'Clipart', 'Product', 'RealWorld']:
    name = '{domain}_{split}'.format(domain=domain, split=split)
    __set[name] = (
      lambda domain=domain, transform=transform: OFFICEHOME(OFFICEHOME_DIR, domain=domain, transform=transform))

def get_dataset(name):
  if name not in __sets:
    raise KeyError('Unknown Dataset: {}'.format(name))
  return __sets[name]()

