import torch.nn as nn
import torch.nn.functional as F
#from .resnet_small import resnet20, resnet26, resnet32
from .resnet_big  import resnet18, resnet50

class MLP(nn.Module):
  def __init__(self, base_model, out_dim):
    super(MLP, self).__init__()
    self.resnet_dict  = {"resnet18": resnet18(),
                         "resnet50": resnet50()}
    resnet =  self._get_basemodel(base_model)
    num_ftrs = resnet.fc.in_features
    self.features = nn.Sequential(*list(resnet.children())[:-1])
    
    #projection mlp
    self.l1 = nn.Linear(num_ftrs, num_ftrs)
    self.l2 = nn.Linear(num_ftrs, out_dim)
    
    
  def _get_basemodel(self, model_name):
    try:
      model = self.resnet_dict[model_name]
      return model
    except:
      raise ("Invalid model name. Check config file.")
  
  def forward(self, x):
    h = self.features(x)
    h = h.squeeze()
    x = self.l1(h)
    x = F.relu(x)
    x = self.l2(x)

    return x
