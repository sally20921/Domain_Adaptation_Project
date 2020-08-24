import os
from pathlib import Path
from torch import optim
from inflection import underscore

optim_dict = {}

def add_optims():
    path = Path(os.path.dirname(__file__))


