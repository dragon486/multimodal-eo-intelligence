"""Loss Functions for Multi-Label Classification and Flood Segmentation."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss to counter class imbalance in satellite segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Multi-Class Dice Loss for flood boundary overlap."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Mask out ignore index
        valid_mask = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets_clean, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(targets_one_hot)

        probs = probs * valid_mask_expanded
        targets_one_hot = targets_one_hot * valid_mask_expanded

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)


class CombinedSegmentationLoss(nn.Module):
    """Composite CrossEntropy + Dice Loss for robust flood segmentation."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = -1):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
