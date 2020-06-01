import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        outputs = torch.sigmoid(outputs)

        numerator = 2 * torch.sum(outputs * targets) + self.smooth
        denominator = torch.sum(outputs ** 2) + torch.sum(targets ** 2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets)

        return soft_dice_loss + bce_loss
