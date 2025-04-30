import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BCEWithLogitsLoss, BCELoss
from utils.metrics import calculate_dice_score, calculate_iou_score


def select_criterion(model_name):
    if model_name=="unet":
        return BCELoss(reduction='mean')
    else:
        return BCEWithLogitsLoss(reduction='mean')

def tversky_loss(preds, targets):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    tversky_idx = tp / (tp + alpha * fp + beta * fn + 1e-8)
    return 1 - tversky_idx.mean()

class BinaryDiceLoss(nn.Module):
    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1e-8
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, preds, targets, use_sigmoid=True):
        assert preds.shape[0] == targets.shape[0], "preds & targets batch size don't match"
        if use_sigmoid:
            preds = torch.sigmoid(preds)

        if self.ignore_index is not None:
            validmask = (targets != self.ignore_index).float()
            preds = preds.mul(validmask)  # can not use inplace for bp
            targets = targets.float().mul(validmask)

        dim0 = preds.shape[0]
        if self.batch_dice:
            dim0 = 1

        preds = preds.contiguous().view(dim0, -1)
        targets = targets.contiguous().view(dim0, -1).float()
        num = 2 * torch.sum(torch.mul(preds, targets), dim=1) + self.smooth
        den = torch.sum(preds.abs() + targets.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

