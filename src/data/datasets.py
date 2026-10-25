"""
PyTorch Dataset and DataLoader for Multimodal Remote Sensing (Sentinel-1 SAR + Sentinel-2 Optical).
Designed as the primary data interface for Person 1's models and Person 2's evaluation pipeline.
"""
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess import load_and_preprocess_pair

class Sen1Floods11Dataset(Dataset):
    """
    Multimodal PyTorch Dataset for paired Sentinel-1 SAR and Sentinel-2 Optical imagery
    with hand-labeled flood segmentation ground truth.
    """
    def __init__(
        self,
        split_csv: str | Path,
        raw_data_dir: str | Path,
        prithvi_bands: bool = False,
        transform: Optional[Callable] = None,
        ignore_missing: bool = True
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.prithvi_bands = prithvi_bands
        self.transform = transform

        self.s1_dir = self.raw_data_dir / "S1Hand"
        self.s2_dir = self.raw_data_dir / "S2Hand"
        self.label_dir = self.raw_data_dir / "LabelHand"

        # Read chip list from official split CSV
        df = pd.read_csv(split_csv, header=None)
        all_chips = [str(x).strip() for x in df[0].tolist()]

        # Filter to only chips where S1, S2, and Label all exist on disk
        self.chips: List[str] = []
        for chip in all_chips:
            s1_file = self.s1_dir / f"{chip}_S1Hand.tif"
            s2_file = self.s2_dir / f"{chip}_S2Hand.tif"
            lbl_file = self.label_dir / f"{chip}_LabelHand.tif"

            if s1_file.exists() and s2_file.exists() and lbl_file.exists():
                self.chips.append(chip)
            elif not ignore_missing:
                raise FileNotFoundError(f"Missing one or more modalities for chip: {chip}")

        if len(self.chips) == 0:
            print(f"[Warning] 0 valid chips found for {split_csv}. Please check data downloads.")

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chip_id = self.chips[idx]
        s1_file = self.s1_dir / f"{chip_id}_S1Hand.tif"
        s2_file = self.s2_dir / f"{chip_id}_S2Hand.tif"
        lbl_file = self.label_dir / f"{chip_id}_LabelHand.tif"

        data = load_and_preprocess_pair(
            s1_path=s1_file,
            s2_path=s2_file,
            label_path=lbl_file,
            prithvi_bands=self.prithvi_bands
        )

        s1_tensor = torch.from_numpy(data["s1"]).float()          # (2, H, W)
        s2_tensor = torch.from_numpy(data["s2"]).float()          # (13 or 6, H, W)
        mask_tensor = torch.from_numpy(data["mask"]).long()       # (H, W)

        item = {
            "chip_id": chip_id,
            "s1": s1_tensor,
            "s2": s2_tensor,
            "mask": mask_tensor,
            "crs": data["crs"]
        }

        if self.transform is not None:
            item = self.transform(item)

        return item

def get_dataloader(
    split_csv: str | Path,
    raw_data_dir: str | Path,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    prithvi_bands: bool = False
) -> DataLoader:
    """Helper factory to create a DataLoader for training or evaluation."""
    dataset = Sen1Floods11Dataset(
        split_csv=split_csv,
        raw_data_dir=raw_data_dir,
        prithvi_bands=prithvi_bands
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
