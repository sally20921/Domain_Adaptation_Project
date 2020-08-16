import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from utils.io_utils import download_url, check_integrity
import zipfile
import h5py
import functools

class OFFICE(data.Dataset):
  def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):
    self.extensions = ['jpg', 'jpeg', 'png']
    domain_root_dir = os.path.join(root, domain, 'images')
    classes = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
                   'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
                   'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer',
                   'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler',
                   'tape_dispenser', 'trash_can']
    class_to_idx = {cls:i for i, cls in enumerate(classes)}
    samples = make_dataset(domain_root_dir, class_to_idx, self.extensions, list_file=list_file)
    if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))
    
    self.root = domain_root_dir
    self.loader = Image.open
    self.domain = domain
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.samples = samples
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
  '''
  returns tuple (sample, target) where target  is class_index of target class
  '''
    path, target = self.samples[index]
    sample = self.loader(path).convert('RGB')
    if self.transfrom is not None:
      sample = self.transform(sample)
    if self.target_transfrom is not None:
      target = self.target_transform(target)

    return sample, target

  def __len__(self):
    return len(self.samples)

  def __repr__(self):
    fmt_str = 'OFFICE Dataset\n'
    return fmt_str

class VISDA(data.Dataset):
  def __init__(self, root, domain, list_file=None, transform=None, target_transform=None):
    self.extensions = ['jpg', 'jpeg', 'png']
    domain_root_dir = os.path.join(root, domain)
    classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant',
                   'skateboard', 'train', 'truck']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    samples = self.make_dataset(domain_root_dir, class_to_idx, self.extensions, list_file=list_file)
    if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n" +
                                "Supported extensions are: " + ",".join(self.extensions)))

    self.root = domain_root_dir
    self.loader= Image.open
    self.domain = domain
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.samples = self.samples
    self.transform = transform
    self.target_transform = target_transform

  
