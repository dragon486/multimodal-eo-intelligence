"""Unit Tests for Foundation Model Forward Pass and Multimodal Fusion."""

import pytest
import torch
from src.models import (
    GeospatialOpticalEncoder,
    SAREncoder,
    MultimodalEarthModel,
)


def test_optical_encoder_shape():
    """Verify 6-band optical encoder produces expected token and CLS dimensions."""
    encoder = GeospatialOpticalEncoder(img_size=224, patch_size=16, in_chans=6, embed_dim=768, depth=2)
    dummy_opt = torch.randn(2, 6, 224, 224)
    tokens, cls_out = encoder(dummy_opt)

    assert tokens.shape == (2, 196, 768), f"Expected (2, 196, 768), got {tokens.shape}"
    assert cls_out.shape == (2, 768), f"Expected (2, 768), got {cls_out.shape}"


def test_sar_encoder_shape():
    """Verify 2-band SAR encoder produces matching token representations."""
    sar_encoder = SAREncoder(img_size=224, patch_size=16, in_chans=2, embed_dim=768, depth=2)
    dummy_sar = torch.randn(2, 2, 224, 224)
    tokens_sar = sar_encoder(dummy_sar)

    assert tokens_sar.shape == (2, 196, 768), f"Expected (2, 196, 768), got {tokens_sar.shape}"


def test_multimodal_classification_forward():
    """Verify multimodal fusion model for BigEarthNet multi-label classification."""
    model = MultimodalEarthModel(
        img_size=224,
        embed_dim=768,
        depth_opt=2,
        depth_sar=2,
        task="classification",
        num_classes=19,
        fusion_method="cross_attention",
    )
    dummy_opt = torch.randn(2, 6, 224, 224)
    dummy_sar = torch.randn(2, 2, 224, 224)
    dummy_mask = torch.zeros(2, 1, 224, 224)

    logits = model(dummy_opt, dummy_sar, dummy_mask)
    assert logits.shape == (2, 19), f"Expected (2, 19), got {logits.shape}"

    # Test predict API
    preds = model.predict(dummy_opt, dummy_sar, dummy_mask)
    assert "logits" in preds and "probabilities" in preds and "predictions" in preds and "uncertainty" in preds
    assert preds["probabilities"].shape == (2, 19)


def test_multimodal_segmentation_forward():
    """Verify multimodal fusion model for Sen1Floods11 pixel-level flood segmentation."""
    model = MultimodalEarthModel(
        img_size=224,
        embed_dim=768,
        depth_opt=2,
        depth_sar=2,
        task="segmentation",
        num_classes=3,
        fusion_method="cross_attention",
    )
    dummy_opt = torch.randn(2, 6, 224, 224)
    dummy_sar = torch.randn(2, 2, 224, 224)

    logits = model(dummy_opt, dummy_sar)
    assert logits.shape == (2, 3, 224, 224), f"Expected (2, 3, 224, 224), got {logits.shape}"


def test_ablation_single_modality():
    """Verify model gracefully supports single-modality inputs (Optical-only / SAR-only)."""
    model = MultimodalEarthModel(
        img_size=224,
        embed_dim=768,
        depth_opt=2,
        depth_sar=2,
        task="classification",
        num_classes=19,
    )
    dummy_opt = torch.randn(2, 6, 224, 224)
    dummy_sar = torch.randn(2, 2, 224, 224)

    logits_opt = model(optical=dummy_opt, sar=None)
    assert logits_opt.shape == (2, 19)

    logits_sar = model(optical=None, sar=dummy_sar)
    assert logits_sar.shape == (2, 19)


if __name__ == "__main__":
    print("Running unit tests...")
    test_optical_encoder_shape()
    print("✓ test_optical_encoder_shape passed")
    test_sar_encoder_shape()
    print("✓ test_sar_encoder_shape passed")
    test_multimodal_classification_forward()
    print("✓ test_multimodal_classification_forward passed")
    test_multimodal_segmentation_forward()
    print("✓ test_multimodal_segmentation_forward passed")
    test_ablation_single_modality()
    print("✓ test_ablation_single_modality passed")
    print("All unit tests passed successfully!")
