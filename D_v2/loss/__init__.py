import os
import inspect 
from pathlib import Path

from torch.nn.modules.loss import _Loss

from inflection import underscore
'''
The inspect module provides several useful functions to help get  information about live  objects such as modules, classes, methods, functions, tracebacks, frame objects, and code objects. 
There are  four main kinds of services provided by this module: type checking, getting source code, inspecting classes and functions, and examining the interpreter stack
'''

loss_dict = {}

def add_loss():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem
        if name != "__init__":
            __import__("{}/{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, '__mro__') and \
                        _Loss in inspect.getmro(member):
                            loss_dict[underscore(str(member.__name__))] = member



def get_loss(args):
    loss = loss_dict[args.loss_name]
    loss = loss.resolve_args(args)
    return loss.to(args.device)

add_loss()
