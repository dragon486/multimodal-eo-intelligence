"""Training script for Flood Segmentation (Experiments E1 - E6).

Features:
- Power-loss and disconnect-resilient checkpointing (auto-resume from last completed epoch)
- Atomic checkpoint saving to prevent corruption on unexpected shutdown
- Offline operation: uses local rasters in data/raw/sen1floods11/ without internet dependency
- Empirical class imbalance weights [0.5566, 4.9183] with ignore_index=-1
- MetricAccumulator for validation water_iou, mean_iou, and water_f1
- AMP FP16 + Gradient Accumulation for 6GB VRAM (RTX 2060)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import get_dataloader, MockMultimodalDataset
from src.models.multimodal_model import MultimodalEarthModel
from src.evaluation.metrics import MetricAccumulator


def train_flood_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mode: str = "multimodal",
    accumulation_steps: int = 4,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training Flood Segmentation", leave=False)
    for i, batch in enumerate(pbar):
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        mask = batch["mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            if mode == "optical_only":
                logits = model(optical=s2, sar=None)
            elif mode == "sar_only":
                logits = model(optical=None, sar=s1)
            else:
                logits = model(optical=s2, sar=s1)

            loss = criterion(logits, mask) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_flood_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = "multimodal",
    num_classes: int = 2,
    ignore_index: int = -1,
) -> Dict[str, Any]:
    model.eval()
    accumulator = MetricAccumulator(num_classes=num_classes, ignore_index=ignore_index)

    for batch in dataloader:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        mask = batch["mask"].to(device)

        if mode == "optical_only":
            logits = model(optical=s2, sar=None)
        elif mode == "sar_only":
            logits = model(optical=None, sar=s1)
        else:
            logits = model(optical=s2, sar=s1)

        preds = logits.argmax(dim=1)
        accumulator.update(preds, mask)

    return accumulator.compute()


def run_experiment(
    exp_id: str = "E5",
    mode: str = "multimodal",
    epochs: int = 10,
    batch_size: int = 2,
    lr: float = 1e-4,
    resume: bool = True,
    use_mock: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 65}")
    print(f"RUNNING FLOOD EXPERIMENT: {exp_id} ({mode.upper()})")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"{'=' * 65}")

    base_dir = Path(__file__).resolve().parents[2]
    splits_dir = base_dir / "data" / "splits"
    raw_dir = base_dir / "data" / "raw" / "sen1floods11"
    checkpoints_dir = base_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Loaders
    if use_mock or not (raw_dir / "S1Hand").exists():
        print("[Notice] Using MockMultimodalDataset conforming to Data Contract for verification.")
        train_ds = MockMultimodalDataset(length=32, img_size=512, in_chans_opt=6, task="segmentation", num_classes=2)
        val_ds = MockMultimodalDataset(length=16, img_size=512, in_chans_opt=6, task="segmentation", num_classes=2)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        in_chans_opt = 6
    else:
        train_loader = get_dataloader(
            split_csv=splits_dir / "flood_train_data.csv",
            raw_data_dir=raw_dir,
            batch_size=batch_size,
            shuffle=True,
            prithvi_bands=True,
        )
        val_loader = get_dataloader(
            split_csv=splits_dir / "flood_valid_data.csv",
            raw_data_dir=raw_dir,
            batch_size=batch_size,
            shuffle=False,
            prithvi_bands=True,
        )
        in_chans_opt = 6

    # 2. Model Initialization
    model = MultimodalEarthModel(
        img_size=512,
        in_chans_opt=in_chans_opt,
        in_chans_sar=2,
        embed_dim=768,
        depth_opt=4,
        depth_sar=2,
        num_heads=8,
        task="segmentation",
        num_classes=2,
        fusion_method="cross_attention",
    ).to(device)

    # 3. Class Imbalance Mitigation (Data Contract Rule 1)
    weights = torch.tensor([0.5566, 4.9183], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Checkpoint paths
    last_ckpt_path = checkpoints_dir / f"last_{exp_id.lower()}_{mode}.pt"
    best_ckpt_path = checkpoints_dir / f"best_{exp_id.lower()}_{mode}.pt"

    start_epoch = 1
    best_water_iou = 0.0

    # 4. Checkpoint Auto-Resume Logic
    if resume and last_ckpt_path.exists():
        print(f"\n[CHECKPOINT DETECTED] Found existing checkpoint: {last_ckpt_path.name}")
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_water_iou = ckpt.get("best_water_iou", 0.0)
            print(f"  --> Resuming training from Epoch {start_epoch}/{epochs} (Previous Best Water IoU: {best_water_iou * 100:.2f}%)")
        except Exception as e:
            print(f"  [Warning] Could not load checkpoint ({e}). Starting fresh from epoch 1.")

    # 5. Training Loop
    if start_epoch > epochs:
        print(f"\n[Completed] Model has already finished all {epochs} epochs. Proceeding directly to evaluation.")
    else:
        for epoch in range(start_epoch, epochs + 1):
            loss = train_flood_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                mode=mode,
                accumulation_steps=2,
            )

            metrics = evaluate_flood_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                mode=mode,
                num_classes=2,
                ignore_index=-1,
            )

            is_best = metrics["water_iou"] > best_water_iou
            if is_best:
                best_water_iou = metrics["water_iou"]

            print(
                f"Epoch [{epoch:>2}/{epochs}] "
                f"Loss: {loss:.4f} | "
                f"Val Water IoU: {metrics['water_iou'] * 100:.2f}% | "
                f"Val Water F1: {metrics['water_f1'] * 100:.2f}% | "
                f"Val mIoU: {metrics['mean_iou'] * 100:.2f}%"
                + ("  [* BEST]" if is_best else "")
            )

            # Atomic save of last checkpoint (resumes smoothly if power turns off)
            last_checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_water_iou": best_water_iou,
                "metrics": metrics,
                "exp_id": exp_id,
                "mode": mode,
            }
            tmp_last = checkpoints_dir / f"last_{exp_id.lower()}_{mode}.tmp.pt"
            torch.save(last_checkpoint_data, tmp_last)
            tmp_last.replace(last_ckpt_path)

            # Save best checkpoint
            if is_best:
                tmp_best = checkpoints_dir / f"best_{exp_id.lower()}_{mode}.tmp.pt"
                torch.save(last_checkpoint_data, tmp_best)
                tmp_best.replace(best_ckpt_path)

    print(f"\nExperiment {exp_id} ({mode.upper()}) complete. Best Validation Water IoU: {best_water_iou * 100:.2f}%")

    # 6. Final Evaluation on Test & Held-Out Bolivia Split
    eval_model_path = best_ckpt_path if best_ckpt_path.exists() else last_ckpt_path
    if eval_model_path.exists():
        ckpt = torch.load(eval_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nLoaded evaluated checkpoint: {eval_model_path.name}")

    test_results = {}
    if (splits_dir / "flood_test_data.csv").exists() and not use_mock:
        test_loader = get_dataloader(
            split_csv=splits_dir / "flood_test_data.csv",
            raw_data_dir=raw_dir,
            batch_size=batch_size,
            shuffle=False,
            prithvi_bands=True,
        )
        test_metrics = evaluate_flood_epoch(model, test_loader, device, mode=mode)
        test_results["test"] = test_metrics
        print("\n" + "=" * 60)
        print(f"OFFICIAL TEST SPLIT EVALUATION ({exp_id} - {mode.upper()})")
        print(f"  Water IoU : {test_metrics['water_iou'] * 100:.2f}%")
        print(f"  Water F1  : {test_metrics['water_f1'] * 100:.2f}%")
        print(f"  Mean IoU  : {test_metrics['mean_iou'] * 100:.2f}%")
        print(f"  Accuracy  : {test_metrics['accuracy'] * 100:.2f}%")
        print("=" * 60)

    if (splits_dir / "flood_bolivia_data.csv").exists() and not use_mock:
        bolivia_loader = get_dataloader(
            split_csv=splits_dir / "flood_bolivia_data.csv",
            raw_data_dir=raw_dir,
            batch_size=batch_size,
            shuffle=False,
            prithvi_bands=True,
        )
        bolivia_metrics = evaluate_flood_epoch(model, bolivia_loader, device, mode=mode)
        test_results["bolivia"] = bolivia_metrics
        print("\n" + "=" * 60)
        print(f"BOLIVIA HELD-OUT GENERALIZATION ({exp_id} - {mode.upper()})")
        print(f"  Water IoU : {bolivia_metrics['water_iou'] * 100:.2f}%")
        print(f"  Water F1  : {bolivia_metrics['water_f1'] * 100:.2f}%")
        print(f"  Mean IoU  : {bolivia_metrics['mean_iou'] * 100:.2f}%")
        print("=" * 60)

    return {
        "exp_id": exp_id,
        "mode": mode,
        "best_val_water_iou": best_water_iou,
        "test_results": test_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="E5", help="Experiment ID: E1, E4, E5")
    parser.add_argument("--mode", type=str, default="multimodal", choices=["multimodal", "optical_only", "sar_only"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoint")
    parser.add_argument("--mock", action="store_true", help="Force synthetic data for testing")
    args = parser.parse_args()

    run_experiment(
        exp_id=args.exp,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=not args.no_resume,
        use_mock=args.mock,
    )
