"""Multimodal Fusion Architectures for Optical and SAR Remote Sensing.

Implements:
1. GatedConcatFusion: Concatenation baseline with learned gating.
2. BidirectionalCrossAttentionFusion: Cross-attention querying SAR structure and optical reflectance.
3. CloudAdaptiveGate: Dynamic downweighting of optical features under cloud degradation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConcatFusion(nn.Module):
    """Early/Feature-level Concatenation with Learned Gating."""

    def __init__(self, opt_dim: int = 768, sar_dim: int = 768, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(opt_dim + sar_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(opt_dim + sar_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_opt: torch.Tensor, z_sar: torch.Tensor) -> torch.Tensor:
        """Args:
            z_opt: (B, N, opt_dim)
            z_sar: (B, N, sar_dim)
        Returns:
            z_fused: (B, N, out_dim)
        """
        combined = torch.cat([z_opt, z_sar], dim=-1)
        projected = self.fusion_proj(combined)
        gate_weight = self.gate(combined)
        return gate_weight * projected + (1.0 - gate_weight) * z_opt


class BidirectionalCrossAttentionFusion(nn.Module):
    """Bidirectional Cross-Attention between Optical and SAR tokens.
    
    Optical queries SAR for cloud-penetrating edge & geometric structure.
    SAR queries Optical for multi-spectral radiometric context.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_opt = nn.LayerNorm(embed_dim)
        self.norm_sar = nn.LayerNorm(embed_dim)

        # Optical queries SAR
        self.cross_opt_to_sar = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # SAR queries Optical
        self.cross_sar_to_opt = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_out = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z_opt: torch.Tensor, z_sar: torch.Tensor) -> torch.Tensor:
        """Args:
            z_opt: Optical tokens (B, N, D)
            z_sar: SAR tokens (B, N, D)
        Returns:
            z_fused: Fused tokens (B, N, D)
        """
        opt_norm = self.norm_opt(z_opt)
        sar_norm = self.norm_sar(z_sar)

        # Optical attends to SAR
        opt_enriched, _ = self.cross_opt_to_sar(
            query=opt_norm,
            key=sar_norm,
            value=sar_norm,
        )
        z_opt_fused = z_opt + opt_enriched

        # SAR attends to Optical
        sar_enriched, _ = self.cross_sar_to_opt(
            query=sar_norm,
            key=opt_norm,
            value=opt_norm,
        )
        z_sar_fused = z_sar + sar_enriched

        # Combined representation
        fused = torch.cat([z_opt_fused, z_sar_fused], dim=-1)
        out = self.norm_out(z_opt_fused + self.mlp(fused))
        return out


class CloudAdaptiveGate(nn.Module):
    """Modulates fused representations based on scene cloud obscuration ratio.
    
    When cloud coverage is high, optical weights are attenuated and SAR backscatter
    is amplified to protect downstream predictions from cloud noise.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.gate_scale = nn.Parameter(torch.ones(1, 1, embed_dim))

    def forward(
        self,
        z_fused: torch.Tensor,
        z_opt: torch.Tensor,
        z_sar: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            z_fused: (B, N, D) tokens from Cross-Attention or Concat
            z_opt: (B, N, D)
            z_sar: (B, N, D)
            cloud_mask: (B, 1, H, W) binary mask (1: cloud, 0: clear)
        """
        if cloud_mask is None:
            return z_fused

        # Compute scene-level cloud fraction: alpha in [0.0, 1.0]
        cloud_ratio = cloud_mask.mean(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
        cloud_ratio = cloud_ratio.squeeze(-1).squeeze(-1)  # (B, 1)

        # Expand to token dimension
        alpha = cloud_ratio.unsqueeze(1)  # (B, 1, 1)

        # Dynamic modulation: as cloud_ratio -> 1, SAR weight increases and optical weight drops
        modulated = z_fused + (alpha * self.gate_scale * z_sar) - (alpha * 0.5 * z_opt)
        return modulated
