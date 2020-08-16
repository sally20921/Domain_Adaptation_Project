import os 
from os.path import join, exists, splitext

import time
from shutil import move, rmtree
import platform

class DatasetPreparer:
  def __init__(self, dataset_name, dataset_root):
    self.dataset_root = dataset_root
    self.dataset_name = dataset_name
    self.dataset_dir = join(dataset_root, dataset_name)

  def download_dataset(self):
    pass
  
  def uncompress(self):
    pass
  
  def refactor(self):
    pass

  def prepare_dataset(self):
    print('Warning: all existing folders will be overwritten')

    self.download_dataset()
    self.uncompress()
    self.refactor()
    print('Completed')


