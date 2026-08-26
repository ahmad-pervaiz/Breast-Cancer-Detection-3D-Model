"""
Dice + BCEWithLogitsLoss combo, weighted, for imbalanced tumor segmentation.

Model outputs raw logits; sigmoid is applied inside the Dice term only (BCE
takes logits directly for numerical stability, per BCEWithLogitsLoss's design).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        target = target.flatten(1)
        intersection = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """total = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss

    forward() returns (total, dice_loss, bce_loss) so the caller can log all three.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        dice_loss = self.dice(logits, target)
        bce_loss = self.bce(logits, target)
        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total, dice_loss.detach(), bce_loss.detach()
