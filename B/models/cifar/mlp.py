import torch.nn as nn
import torch.nn.functional as F
from .resnet_small import resnet20, resnet26, resnet32
from .resnet_big  import resnet18, resnet50

class MLP(nn.Module):
  def __init__(self, base_model):
    super(MLP, self).__init__()
    self.resnet_dict  = {"resnet20": resnet20(),
                         "resnet26": resnet26(),
                         "resnet32": resnet32(),
                         "resnet18": resnet18(),
                         "resnet50": resnet50()}
    resnet =  self._get_basemodel(base_model)
    
  def _get_basemodel(self, model_name):
    try:
      model = self.resnet_dict[model_name]
      return model
    except:
      rase ("Invalid model name. Check config file.")
  
 
