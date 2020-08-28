from collections import defaultdict
import torch

from utils import *

from .preprocess_image import preprocess_images

import os
import re
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, DataLoader

class MultiModalData(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode

    def __getitem__(self, idx):
        

def get_iterator(args):
    iters = load_data(args)
    print("Data laoding done")
    return iters 


