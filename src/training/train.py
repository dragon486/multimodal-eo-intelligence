"""PyTorch Training Engine optimized for 6 GB VRAM (RTX 2060).

Features:
- Automatic Mixed Precision (AMP FP16)
- Gradient Accumulation (simulates large batch size on 6GB VRAM)
- Multi-Task Support (Classification & Flood Segmentation)
"""

from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 4,
    use_amp: bool = True,
) -> float:
    """Runs one training epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for i, batch in enumerate(pbar):
        # Extract inputs matching Data Contract
        opt = batch.get("optical").to(device) if "optical" in batch else None
        sar = batch.get("sar").to(device) if "sar" in batch else None
        cloud_mask = batch.get("cloud_mask").to(device) if "cloud_mask" in batch else None
        targets = batch["target"].to(device)

        # Forward pass with AMP FP16
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            logits = model(opt, sar, cloud_mask)
            loss = criterion(logits, targets) / accumulation_steps

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Step optimizer every accumulation_steps batches
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Runs validation evaluation."""
    model.eval()
    total_loss = 0.0

    for batch in dataloader:
        opt = batch.get("optical").to(device) if "optical" in batch else None
        sar = batch.get("sar").to(device) if "sar" in batch else None
        cloud_mask = batch.get("cloud_mask").to(device) if "cloud_mask" in batch else None
        targets = batch["target"].to(device)

        logits = model(opt, sar, cloud_mask)
        loss = criterion(logits, targets)
        total_loss += loss.item()

    return total_loss / len(dataloader)
