import math
import os
from collections import defaultdict

import PIL 
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn, utils
from torchvision import models, datasets, transforms
from utils import *

def preprocess_image(args):
    print('Loading visual')

