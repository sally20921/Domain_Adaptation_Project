import torch.nn as nn
import torchvision

class I(nn.Module):
    def __init__(self):
        super(I, self).__init__()

    def forward(self,x):
        return x

class SimCLR(nn.Module):
    def __init__(self, args):
        super(SimCLR, self).__init__()

        self.args = args
        self.encoder = self.get_resnet_model(args['base']['resnet'])
        self.num_features = self.encoder.fc.in_features

        # remove fully connected layer
        self.encoder.fc = I()

        self.projector = nn.Sequential(
                nn.Linear(self.num_features, self.num_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.num_features, args['base']['projection_dim'], bias=False)
        )

    @staticmethod
    def get_resnet_model(name):
        if name == 'resnet18':
            model = torchvision.models.resnet18()
        elif name == 'resnet50':
            model = torchvision.models.resnet50()
        else:
            raise KeyError('invalid resnet name: {}'.format(name))

        return model

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        if self.args['base']['normalize']:
            z = nn.functional.normalize(z, dim=1)
        return h,z
