from __future__ import print_function, division, absolute_import
import math
from six.moves.urllib.request import urlretrieve

import torch
from PIL import Image
from tqdm import tqdm

def load_imagenet_classes(path_synsets='./imagenet_synsets.txt',
                          path_classes='./imagenet_classes.txt'):
  with open(path_synsets, 'r') as f:
    synsets = f.readlines()

  synsets = [x.strip() for x in synsets]
  splits = [line.split(' ') for line in synsets]
  key_to_classname = {spl[0]: ' '.join(sp[1:]) for spl in splits}

  with open(path_classes, 'r') as f:
    class_id_to_key = f.readlines()

  class_id_to_key = [x.strip() for x in class_id_to_key]

  cid_to_cname =[]
  for i in range(len(class_id_to_key)):
    key = class_id_to_key[i]
    cname = key_to_classname[key]
    cid_to_cname.append(cname)

  return cid_to_cname


