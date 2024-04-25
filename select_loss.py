import torch
from torch import nn

class Select_Loss(nn.Module):
    def __init__(self,args):
        super(Select_Loss,self).__init__()
        self.args = args
        self.l1loss  = nn.L1Loss()

    def forward(self,sr,gt):
        l1loss = self.l1loss(sr,gt)
        loss = l1loss 
        return loss