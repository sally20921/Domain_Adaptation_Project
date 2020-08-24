import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class SimCLR(nn.Module):

