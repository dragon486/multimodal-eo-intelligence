"""Inference functions and Handoff Interface for Person 2 and Dashboard.

Implements:
1. predict_chip(s1_tensor, s2_tensor): Exact signature requested by Person 2 in PERSON_1_HANDOFF_PACKAGE.md.
2. MultimodalPredictor: High-level inference engine for model checkpoints.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from src.models.multimodal_model import MultimodalEarthModel

# Global model instance for fast repeated dashboard calls
_GLOBAL_MODEL: Optional[MultimodalEarthModel] = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_or_load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    task: str = "segmentation",
    num_classes: int = 2,
    in_chans_opt: int = 6,
) -> MultimodalEarthModel:
    """Loads model singleton onto GPU/CPU."""
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None:
        # Default to trained multimodal checkpoint if available
        if checkpoint_path is None:
            default_ckpt = Path(__file__).resolve().parents[2] / "checkpoints" / "best_e5_multimodal.pt"
            if default_ckpt.exists():
                checkpoint_path = default_ckpt

        depth_opt = 4
        depth_sar = 2
        state_dict = None

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Auto-detect layer depths from checkpoint
            opt_blocks = [int(k.split(".")[2]) for k in state_dict.keys() if k.startswith("optical_encoder.blocks.") and ".norm1.weight" in k]
            sar_blocks = [int(k.split(".")[2]) for k in state_dict.keys() if k.startswith("sar_encoder.blocks.") and ".norm1.weight" in k]
            if opt_blocks:
                depth_opt = max(opt_blocks) + 1
            if sar_blocks:
                depth_sar = max(sar_blocks) + 1

        model = MultimodalEarthModel(
            img_size=512,
            in_chans_opt=in_chans_opt,
            in_chans_sar=2,
            depth_opt=depth_opt,
            depth_sar=depth_sar,
            task=task,
            num_classes=num_classes,
            fusion_method="cross_attention",
        ).to(_DEVICE)

        if state_dict is not None:
            model.load_state_dict(state_dict)
            print(f"[Model] Successfully loaded weights (depth_opt={depth_opt}, depth_sar={depth_sar}) from {checkpoint_path}")

        model.eval()
        _GLOBAL_MODEL = model

    return _GLOBAL_MODEL


def predict_chip(
    s1_tensor: Union[torch.Tensor, np.ndarray],
    s2_tensor: Union[torch.Tensor, np.ndarray],
    checkpoint_path: Optional[str] = None,
) -> np.ndarray:
    """Standardized inference function for Person 2's dashboard and evaluation pipeline.
    
    Args:
        s1_tensor: (2, 512, 512) or (1, 2, 512, 512) SAR tensor
        s2_tensor: (6, 512, 512) or (13, 512, 512) Optical tensor
        checkpoint_path: Optional path to trained .pt checkpoint
    Returns:
        binary_mask: (512, 512) integer numpy array with class predictions (0=land, 1=water)
    """
    # Convert numpy to torch if needed
    if isinstance(s1_tensor, np.ndarray):
        s1_tensor = torch.from_numpy(s1_tensor)
    if isinstance(s2_tensor, np.ndarray):
        s2_tensor = torch.from_numpy(s2_tensor)

    # Ensure batch dimension: (B=1, C, H, W)
    if s1_tensor.dim() == 3:
        s1_tensor = s1_tensor.unsqueeze(0)
    if s2_tensor.dim() == 3:
        s2_tensor = s2_tensor.unsqueeze(0)

    # If 13-band Sentinel-2, extract 6 Prithvi bands [B2, B3, B4, B8A, B11, B12]
    if s2_tensor.shape[1] == 13:
        prithvi_indices = [1, 2, 3, 8, 11, 12]
        s2_tensor = s2_tensor[:, prithvi_indices, :, :]

    s1_tensor = s1_tensor.float().to(_DEVICE)
    s2_tensor = s2_tensor.float().to(_DEVICE)

    model = get_or_load_model(checkpoint_path=checkpoint_path, in_chans_opt=s2_tensor.shape[1])

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
            logits = model(optical=s2_tensor, sar=s1_tensor)
            preds = logits.argmax(dim=1)  # (1, 512, 512)

    return preds.squeeze(0).cpu().numpy().astype(np.int64)


if __name__ == "__main__":
    print("Testing predict_chip interface...")
    dummy_s1 = np.random.rand(2, 512, 512).astype(np.float32)
    dummy_s2 = np.random.rand(13, 512, 512).astype(np.float32)

    pred_mask = predict_chip(dummy_s1, dummy_s2)
    print(f"Prediction successful! Output mask shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
    print(f"Unique predicted classes: {np.unique(pred_mask)}")
