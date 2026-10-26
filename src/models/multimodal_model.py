"""End-to-End Multimodal Earth Observation Architecture.

Integrates:
- Optical Geospatial ViT (Prithvi-EO-2.0 / 6 bands)
- Synthetic Aperture Radar (SAR) Encoder (2 bands VV/VH)
- Multimodal Fusion Engine (Cross-Attention / Gated Concat)
- Cloud-Adaptive Gating
- Downstream Classification and Segmentation Heads
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .foundation_encoder import GeospatialOpticalEncoder, SAREncoder
from .fusion import BidirectionalCrossAttentionFusion, GatedConcatFusion, CloudAdaptiveGate
from .task_heads import ClassificationHead, SegmentationDecoder


class MultimodalEarthModel(nn.Module):
    """Unified Multimodal Remote Sensing Foundation Architecture."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans_opt: int = 6,
        in_chans_sar: int = 2,
        embed_dim: int = 768,
        depth_opt: int = 12,
        depth_sar: int = 4,
        num_heads: int = 12,
        fusion_method: str = "cross_attention",  # "cross_attention" or "concat"
        task: str = "classification",  # "classification" or "segmentation"
        num_classes: int = 19,
        use_cloud_gate: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.fusion_method = fusion_method
        self.use_cloud_gate = use_cloud_gate

        # 1. Encoders
        self.optical_encoder = GeospatialOpticalEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans_opt,
            embed_dim=embed_dim,
            depth=depth_opt,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.sar_encoder = SAREncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans_sar,
            embed_dim=embed_dim,
            depth=depth_sar,
            num_heads=8,
            dropout=dropout,
        )

        # 2. Multimodal Fusion Module
        if fusion_method == "cross_attention":
            self.fusion = BidirectionalCrossAttentionFusion(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=dropout,
            )
        else:
            self.fusion = GatedConcatFusion(
                opt_dim=embed_dim,
                sar_dim=embed_dim,
                out_dim=embed_dim,
                dropout=dropout,
            )

        # 3. Cloud-Adaptive Gating
        self.cloud_gate = CloudAdaptiveGate(embed_dim=embed_dim) if use_cloud_gate else None

        # 4. Downstream Task Heads
        if task == "classification":
            self.head = ClassificationHead(
                embed_dim=embed_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif task == "segmentation":
            self.head = SegmentationDecoder(
                embed_dim=embed_dim,
                num_classes=num_classes,
                patch_size=patch_size,
                img_size=img_size,
            )
        else:
            raise ValueError(f"Unknown task: {task}")

    def freeze_backbones(self):
        """Freeze optical and SAR foundation encoders (Level 1 training)."""
        self.optical_encoder.freeze()
        for p in self.sar_encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze encoders for end-to-end fine-tuning (Level 2 training)."""
        self.optical_encoder.unfreeze()
        for p in self.sar_encoder.parameters():
            p.requires_grad = True

    def forward(
        self,
        optical: Optional[torch.Tensor] = None,
        sar: Optional[torch.Tensor] = None,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass supporting multimodal, optical-only, and SAR-only modes.
        
        Args:
            optical: (B, 6, H, W)
            sar: (B, 2, H, W)
            cloud_mask: (B, 1, H, W) optional cloud mask
        Returns:
            logits: (B, num_classes) for classification OR (B, num_classes, H, W) for segmentation
        """
        # Ablation mode: Optical-only
        if optical is not None and sar is None:
            tokens_opt, cls_opt = self.optical_encoder(optical)
            if self.task == "classification":
                return self.head(tokens_opt, cls_token=cls_opt)
            return self.head(tokens_opt)

        # Ablation mode: SAR-only
        if optical is None and sar is not None:
            tokens_sar = self.sar_encoder(sar)
            if self.task == "classification":
                return self.head(tokens_sar)
            return self.head(tokens_sar)

        # Multimodal mode: Both optical and SAR available
        if optical is not None and sar is not None:
            tokens_opt, cls_opt = self.optical_encoder(optical)
            tokens_sar = self.sar_encoder(sar)

            # Fuse tokens
            z_fused = self.fusion(tokens_opt, tokens_sar)

            # Apply cloud-adaptive gating if mask provided
            if self.cloud_gate is not None and cloud_mask is not None:
                z_fused = self.cloud_gate(z_fused, tokens_opt, tokens_sar, cloud_mask)

            if self.task == "classification":
                return self.head(z_fused)
            return self.head(z_fused)

        raise ValueError("At least one modality (optical or SAR) must be provided.")

    @torch.no_grad()
    def predict(
        self,
        optical: Optional[torch.Tensor] = None,
        sar: Optional[torch.Tensor] = None,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference API for Person 2 and Dashboard integration.
        
        Returns:
            Dictionary containing:
            - "logits": Raw model outputs
            - "probabilities": Sigmoid / Softmax probabilities
            - "predictions": Discrete class predictions or binary thresholded masks
            - "uncertainty": Prediction entropy / variance
        """
        self.eval()
        logits = self.forward(optical, sar, cloud_mask)

        if self.task == "classification":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
            return {
                "logits": logits,
                "probabilities": probs,
                "predictions": preds,
                "uncertainty": entropy.mean(dim=-1),
            }
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            return {
                "logits": logits,
                "probabilities": probs,
                "predictions": preds,
                "uncertainty": entropy,
            }
