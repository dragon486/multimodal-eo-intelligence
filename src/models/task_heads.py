"""Task-Specific Prediction Heads for Land-Cover and Flood Segmentation.

Implements:
1. ClassificationHead: Multi-label classification for BigEarthNet (19 classes).
2. SegmentationDecoder: Lightweight convolutional decoder for Sen1Floods11 pixel-level segmentation.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Multi-Label Land-Cover Classification Head."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 19,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, tokens: torch.Tensor, cls_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cls_token is not None:
            feat = cls_token
        else:
            feat = tokens.mean(dim=1)  # GAP over patches

        return self.classifier(feat)


class SegmentationDecoder(nn.Module):
    """Pixel-level Semantic Segmentation Decoder for Flood Inundation Mapping.
    
    Dynamically infers grid size from token count N, applies transposed convolutions,
    and interpolates to the target spatial resolution (e.g. 512x512 or 224x224).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,  # 0: Non-water, 1: Flood water (or 3 for multi-class)
        patch_size: int = 16,
        img_size: int = 512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        # Reshape projection
        self.proj = nn.Conv2d(embed_dim, 256, kernel_size=1)

        # Progressive upsampling blocks (scale 16x total = 2^4)
        self.up1 = self._make_up_block(256, 128)
        self.up2 = self._make_up_block(128, 64)
        self.up3 = self._make_up_block(64, 32)
        self.up4 = self._make_up_block(32, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_up_block(self, in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor, target_size: Optional[int] = None) -> torch.Tensor:
        """Args:
            tokens: (B, N, embed_dim) where N = H_g * W_g
            target_size: Optional output resolution (defaults to self.img_size)
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, N, D = tokens.shape
        H_g = int(math.isqrt(N))
        W_g = H_g

        # (B, N, D) -> (B, D, H_g, W_g)
        x = tokens.transpose(1, 2).reshape(B, D, H_g, W_g)
        x = self.proj(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        logits = self.head(x)  # shape is (B, num_classes, H_g * 16, W_g * 16)

        out_size = target_size or self.img_size
        if logits.shape[-1] != out_size or logits.shape[-2] != out_size:
            logits = F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)

        return logits
