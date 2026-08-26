"""
Dice + BCEWithLogitsLoss combo, weighted, for imbalanced tumor segmentation.

Model outputs raw logits; sigmoid is applied inside the Dice term only (BCE
takes logits directly for numerical stability, per BCEWithLogitsLoss's design).
"""
from typing import Optional

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

    pos_weight: passed straight through to BCEWithLogitsLoss - multiplies the
    loss on positive (tumor) pixels, trading precision for recall as it
    increases. None (default) = neutral/off. Tried at 2.0 in an experiment
    (see improving_model.md changelog) - too aggressive, tanked precision
    for a small recall gain and a net loss on val_dice. Left configurable
    here (rather than removed) so a milder value can be tried later without
    re-adding this from scratch.
    """

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 pos_weight: Optional[float] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        pw = torch.tensor(pos_weight) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        dice_loss = self.dice(logits, target)
        bce_loss = self.bce(logits, target)
        total = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total, dice_loss.detach(), bce_loss.detach()
