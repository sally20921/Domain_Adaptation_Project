import  torch
import torch.nn as nn
import numpy as np

class NT_Xent(nn.Module):
    def __init__(self, device, bsz, temp):
        super(NT_Xent, self).__init__()
        self.bsz = bsz
        self.temp = temp
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity = nn.CosineSimilarity(dim=2)
        self.mask = self._mask().type(torch.bool)

    def _mask(self):
        diag = np.eye(2*self.bsz)
        l1 = np.eye((2*self.bsz),2*self.bsz, k=-self.bsz)
        l2 = np.eye((2*self.bsz), 2*self.bsz, k=self.bsz)
        mask = torch.from_numpy((diag+l1+l2))
        mask = (1-mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, z_i, z_j):
        temp = torch.cat([z_j, z_i], dim=0)
        matrix = self.similarity(temp, temp)
        pos1 = torch.diag(matrix, self.bsz)
        pos2 = torch.diag(matrix, -self.bsz)
        positives = torch.cat([pos1, pos2]).view(2*self.bsz, 1)
        negatives = matrix[self._mask].view(2*self.bsz, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits = logits/self.temp
        labels = torch.zeros(2*self.bsz).to(self.device).long()
        loss = self.criterion(logits, labels)
        loss = loss/(2*self.bsz)
        return loss
