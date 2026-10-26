"""PyTorch Dataset and DataLoader for Multimodal Remote Sensing.

Supports:
1. Sen1Floods11Dataset: Paired Sentinel-1 SAR + Sentinel-2 Optical hand-labeled flood dataset.
2. MockMultimodalDataset: Synthetic dataset for fast verification and unit tests.
3. get_dataloader: Standardized factory for training and validation loaders.
"""

from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess import load_and_preprocess_pair


class Sen1Floods11Dataset(Dataset):
    """Multimodal PyTorch Dataset for paired Sentinel-1 SAR and Sentinel-2 Optical imagery
    with hand-labeled flood segmentation ground truth.
    """

    def __init__(
        self,
        split_csv: Union[str, Path],
        raw_data_dir: Union[str, Path],
        prithvi_bands: bool = False,
        transform: Optional[Callable] = None,
        ignore_missing: bool = True,
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
            # Strip file extension if present in CSV line
            clean_chip = chip.replace("_S1Hand.tif", "").replace(".tif", "").strip()
            s1_file = self.s1_dir / f"{clean_chip}_S1Hand.tif"
            s2_file = self.s2_dir / f"{clean_chip}_S2Hand.tif"
            lbl_file = self.label_dir / f"{clean_chip}_LabelHand.tif"

            if s1_file.exists() and s2_file.exists() and lbl_file.exists():
                self.chips.append(clean_chip)
            elif not ignore_missing:
                raise FileNotFoundError(f"Missing one or more modalities for chip: {clean_chip}")

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
            prithvi_bands=self.prithvi_bands,
        )

        s1_tensor = torch.from_numpy(data["s1"]).float()       # (2, H, W)
        s2_tensor = torch.from_numpy(data["s2"]).float()       # (13 or 6, H, W)
        mask_tensor = torch.from_numpy(data["mask"]).long()    # (H, W)

        item = {
            "chip_id": chip_id,
            "s1": s1_tensor,
            "s2": s2_tensor,
            "mask": mask_tensor,
            # Aliases matching both Person 1 and Person 2 conventions
            "sar": s1_tensor,
            "optical": s2_tensor,
            "target": mask_tensor,
        }

        if self.transform is not None:
            item = self.transform(item)

        return item


def get_dataloader(
    split_csv: Union[str, Path],
    raw_data_dir: Union[str, Path],
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    prithvi_bands: bool = False,
) -> DataLoader:
    """Helper factory to create a DataLoader for training or evaluation."""
    dataset = Sen1Floods11Dataset(
        split_csv=split_csv,
        raw_data_dir=raw_data_dir,
        prithvi_bands=prithvi_bands,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class MockMultimodalDataset(Dataset):
    """Synthetic dataset conforming strictly to the Shared Data Contract.
    
    Used for unit testing, VRAM benchmarking, and verifying forward passes.
    """

    def __init__(
        self,
        length: int = 64,
        img_size: int = 512,
        in_chans_opt: int = 6,
        task: str = "segmentation",
        num_classes: int = 2,
        simulate_clouds: bool = True,
    ):
        self.length = length
        self.img_size = img_size
        self.in_chans_opt = in_chans_opt
        self.task = task
        self.num_classes = num_classes
        self.simulate_clouds = simulate_clouds

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        optical = torch.rand(self.in_chans_opt, self.img_size, self.img_size, dtype=torch.float32)
        sar = torch.rand(2, self.img_size, self.img_size, dtype=torch.float32)

        if self.simulate_clouds and np.random.rand() > 0.5:
            cloud_mask = (torch.rand(1, self.img_size, self.img_size) > 0.7).float()
        else:
            cloud_mask = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        if self.task == "classification":
            target = (torch.rand(19) > 0.8).float()
        else:
            # Sen1Floods11 classes: 0 = non-water, 1 = flood water, -1 = ignore
            target = torch.randint(0, self.num_classes, (self.img_size, self.img_size), dtype=torch.long)
            # Add some ignored nodata/cloud pixels
            if self.simulate_clouds:
                ignore_mask = torch.rand(self.img_size, self.img_size) > 0.85
                target[ignore_mask] = -1

        return {
            "chip_id": f"Mock_{idx:04d}",
            "s1": sar,
            "s2": optical,
            "sar": sar,
            "optical": optical,
            "cloud_mask": cloud_mask,
            "mask": target,
            "target": target,
        }
