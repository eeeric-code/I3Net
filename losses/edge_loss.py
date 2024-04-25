import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        c=1 # channel, RGBimg=3
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(c,1,1,1) 
        # if torch.cuda.is_available():
        #     self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, sr, gt):
        sr = sr.permute(3,0,1,2).unsqueeze(2) #[slice,bz,h,w]
        gt = gt.permute(3,0,1,2).unsqueeze(2)
        self.kernel = self.kernel.cuda(sr.device)
        edge_loss = 0
        for input,target in zip(sr,gt): 
            loss = self.loss(self.laplacian_kernel(input), self.laplacian_kernel(target))
            edge_loss += loss
        return edge_loss
