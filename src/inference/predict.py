"""Inference script for Multimodal Remote Sensing Predictions."""

import argparse
from typing import Dict, Optional
import torch
import numpy as np

from src.models.multimodal_model import MultimodalEarthModel


def run_inference(
    model_checkpoint: Optional[str] = None,
    task: str = "classification",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, np.ndarray]:
    """Runs inference on a paired optical-SAR scene."""
    device_obj = torch.device(device)
    model = MultimodalEarthModel(
        img_size=224,
        task=task,
        num_classes=19 if task == "classification" else 3,
    ).to(device_obj)

    if model_checkpoint:
        checkpoint = torch.load(model_checkpoint, map_location=device_obj)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {model_checkpoint}")

    # Generate synthetic sample conforming to Data Contract
    dummy_opt = torch.rand(1, 6, 224, 224, device=device_obj)
    dummy_sar = torch.rand(1, 2, 224, 224, device=device_obj)
    dummy_mask = torch.zeros(1, 1, 224, 224, device=device_obj)

    results = model.predict(dummy_opt, dummy_sar, dummy_mask)
    return {k: v.cpu().numpy() for k, v in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "segmentation"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    output = run_inference(model_checkpoint=args.checkpoint, task=args.task)
    print("Inference completed successfully!")
    print(f"Prediction keys: {list(output.keys())}")
    for k, v in output.items():
        print(f"  {k}: shape {v.shape}")
