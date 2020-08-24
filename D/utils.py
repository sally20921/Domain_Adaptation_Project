from contextlib import contextmanager
from datetime import datetime

import os
import sys
import json
import pickle
import re

import six
import numpy as np
import torch 
from config import log_keys

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path, **kwargs):
    with open(path, 'w') as f:
        json.dump(data, f, **kwargs)

def get_dirname_from_args(args):
    dirname=''
    for key in sorted(log_keys):
        dirname +=  '_'
        dirname += key
        dirname += '_'
        dirname += str(args[key])

    return dirname[1:]

def get_now():
    now = datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S')


