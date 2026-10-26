"""Geospatial Foundation Model Encoders for Optical and SAR Remote Sensing.

Implements encoders for:
1. Optical Foundation Encoder (Prithvi-EO-2.0 / 6-band or 13-band Geospatial ViT)
2. Synthetic Aperture Radar (SAR) Encoder (2-band VV/VH specialized encoder)
With dynamic 2D bicubic positional interpolation to support both 224x224 and 512x512 chips.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_pos_embed(pos_embed: torch.Tensor, num_patches_new: int, has_cls: bool = True) -> torch.Tensor:
    """Interpolates positional embedding to match a different spatial patch count."""
    N = pos_embed.shape[1] - (1 if has_cls else 0)
    if N == num_patches_new:
        return pos_embed

    cls_pos = pos_embed[:, :1] if has_cls else None
    patch_pos = pos_embed[:, 1:] if has_cls else pos_embed

    dim = pos_embed.shape[-1]
    gs_old = int(math.isqrt(N))
    gs_new = int(math.isqrt(num_patches_new))

    patch_pos = patch_pos.reshape(1, gs_old, gs_old, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, dim)

    if has_cls and cls_pos is not None:
        return torch.cat((cls_pos, patch_pos), dim=1)
    return patch_pos


class PatchEmbed2D(nn.Module):
    """2D Image to Patch Embedding for multi-spectral remote sensing tensors."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 6,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer Block with LayerNorm, Multi-Head Self-Attention, and MLP."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GeospatialOpticalEncoder(nn.Module):
    """Optical Foundation Vision Transformer.
    
    Accepts standardized 6-band Sentinel-2 / Landsat HLS tensor or 13-band Sentinel-2 tensor.
    Prithvi HLS 6 bands: [B02, B03, B04, B8A, B11, B12]
    Output: Patch token embeddings (B, num_patches, embed_dim) and pooled representation.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 6,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            tokens, cls_out = self._forward_2d(x)
            return tokens, cls_out
        return self._forward_2d(x)

    def _forward_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        num_patches = x.shape[1]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        pos_embed = interpolate_pos_embed(self.pos_embed, num_patches, has_cls=True)
        x = self.pos_drop(x + pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        patch_tokens = x[:, 1:]
        return patch_tokens, cls_out


class SAREncoder(nn.Module):
    """Synthetic Aperture Radar (SAR) Encoder.
    
    Accepts 2-band Sentinel-1 GRD tensor:
    Channels: [VV, VH] backscatter (calibrated dB normalized)
    Projects SAR structural features to token representation space.
    """

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        in_chans: int = 2,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        num_patches = x.shape[1]
        pos_embed = interpolate_pos_embed(self.pos_embed, num_patches, has_cls=False)
        x = self.pos_drop(x + pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
