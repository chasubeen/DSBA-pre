import torch
import torch.nn as nn
from models.DLinear import DLinear
from layers.RevIN import RevIN


class DLinear_RevIN(nn.Module):
    def __init__(self, configs):
        super(DLinear_RevIN, self).__init__()
        self.revin = RevIN(configs.dim_in)
        self.model = DLinear(configs)  # 기존 DLinear model을 backbone으로 활용

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # [B, L, C] -> Normalize across sequence
        x_enc = self.revin(x_enc, mode='norm')  # normalize
        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        out = self.revin(out, mode='denorm')    # denormalize
        return out
