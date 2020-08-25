import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
