import numpy as np

import torch

from torch import nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
import math
import torch.cuda

__all__ = ["litemono"]


class LiteMono(nn.Module):
    
    def __init__(self, in_chans=3, num_classes=1000, head_init_scale=1., ** kwargs):
        super().__init__()

   
        self.norm = nn.LayerNorm(dims[-2], eps=1e-6)  # Final norm layer
        self.head = nn.Linear(dims[-2], num_classes)

        self.apply(self._init_weights)
        self.head_dropout = nn.Dropout()
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


    def forward_features(self, x):
        # x = (x - 0.45) / 0.225  don't do the normalization because the training script already does that! so please keep this line commented.

        for i in range(1, 3):
            ...

        return self.norm(x.mean([-2, -1]))  # Global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

