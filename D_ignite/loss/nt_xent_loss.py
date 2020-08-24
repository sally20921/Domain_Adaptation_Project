import torch
import torch.nn as nn
import numpy as np
'''
We randomly sample a minibatch of N examples.
Define the contrastive prediction task resulting in 2N data points.
Treat the other 2(N-1) augmented examples as negative examples.
'''
class NTXentLoss(torch.nn.Module):
    def __init__(self, bsz, temp):
        super(NTXentLoss, self).__init__()
        self.bsz = bsz
        self.temp = temp

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def  mask_correlated_samples(self):
        diag = np.eye(2*self.bsz)

    def forward(self, z_i, z_j):

