'''
resnet for cifar dataset (32x32)
A. ResNet-20,26,32(feature dimension 64)
B. ResNet-18,50(feature dimension 512/2048)
'''

import torch.nn as nn
import math

__all__ = ['resnet20', 'resnet26', 'resnet32']

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,*args, **kwargs):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    if inplanes != planes:
      self.downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride), 
        nn.BatchNorm2d(planes)
      )
    else:
      self.downsample = lambda x: x
    
    self.stride = stride

  def forward(self, x):
    residual = self.downsample(x)
   
    out =  self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += residual 
    out = self.relu(out)

    return out  

class Bottleneck(nn.Module):
  expansion =  4
  
  def __init__(self, inplanes, planes, stride=1,  downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = conv1x1(inplanes, planes)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes, stride)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = conv1x1(planes, planes * self.expansion)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion)
    self.relu =  nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x
    
    out =  self.conv1(x)
    out =  self.bn1(out)
    out =  self.relu(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
   
    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample  is not None:
      identity = self.downsample(x)

    out +=  identity
    out  = self.relu(out)

    return out 

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=100, zero_init_residual = False):
    super(ResNet, self).__init__()
    self.inplanes = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu =  nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512*block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
     
      if zero_init_residual:
        for m in self.modules():
          if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
          elif isinstance(m, BasicBlock):
            nn.init.constant_(m, bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = stride != 1
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(block(self.inplanes, planes, 1))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
 
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
   
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x 

class ResNetSmall(nn.Module):
  def __init__(self, block, num_blocks, num_classes=100):
    super(ResNetSmall, self).__init__()

    self.in_planes = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,  bias=False)
    self.bn1  = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    self.linear = nn.Linear(256 * block.expansion, num_classes)
    self.n_channels = [16,32,64]

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride]+[1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x, is_feat=False, use_relu=True):
    out = self.conv1(x)
    out = self.bn1(out)
    if use_relu:
      out = F.relu(out)
    feat1  = self.layer1(out)
    if use_relu:
      feat1 = F.relu(feat1)
    feat2 = self.layer2(feat1)
    if use_relu:
      feat2 = F.relu(feat2)
    feat3 =  self.layer3(feat2)

    #the  last relu is always included
    feat3 = F.relu(feat3)
    pool = F.avg_pool2d(feat3, 4)
    pool = pool.view(pool.size(0), -1)
    out = self.linear(pool)

    if  is_feat:
      return [feat1, feat2, feat3], pool, out

    return out 


def resnet20(**kwargs):
  return ResNetSmall(BasicBlock, [3,3,3], **kwargs)

def resnet26(**kwargs):
  return ResNetSmall(BasicBlock, [4,4,4], **kwargs)

def resnet32(**kwargs):
  return ResNetSmall(BasicBlock, [5,5,5], **kwargs)
