"""
Class Imbalance and Loss Weighting Utility.
Analyzes pixel-level distribution across dataset chips to compute
recommended loss weights (inverse frequency / effective number of samples)
to ensure Person 1 does not suffer from majority-class collapse.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.data.preprocess import read_tiff

def compute_class_imbalance(label_dir: str | Path, max_samples: int = 50) -> Dict[str, Any]:
    """
    Scans ground truth masks in label_dir, counts pixels for:
      -1 : Ignore / Nodata
       0 : Non-water (Land)
       1 : Water (Flood)
    Returns pixel statistics, positive class percentage, and recommended loss weights.
    """
    label_dir = Path(label_dir)
    mask_files = list(label_dir.glob("*_LabelHand.tif"))[:max_samples]

    if not mask_files:
        print(f"[!] No label files found in {label_dir}")
        return {}

    total_ignore = 0
    total_land = 0
    total_water = 0

    print(f">>> Scanning {len(mask_files)} ground truth masks for class imbalance...")

    for f in mask_files:
        arr = read_tiff(f)
        if arr.ndim == 3:
            arr = arr[0]
        
        total_ignore += int((arr == -1).sum())
        total_land += int((arr == 0).sum())
        total_water += int((arr == 1).sum())

    total_valid = total_land + total_water
    total_pixels = total_valid + total_ignore

    pct_land = (total_land / total_valid * 100) if total_valid > 0 else 0
    pct_water = (total_water / total_valid * 100) if total_valid > 0 else 0
    pct_ignore = (total_ignore / total_pixels * 100) if total_pixels > 0 else 0

    # Inverse frequency weights
    # w_c = N_valid / (2 * N_c)
    w_land = float(total_valid / (2.0 * max(total_land, 1)))
    w_water = float(total_valid / (2.0 * max(total_water, 1)))

    # Pos weight for BCEWithLogitsLoss: N_neg / N_pos
    pos_weight = float(total_land / max(total_water, 1))

    stats = {
        "num_files_scanned": len(mask_files),
        "total_valid_pixels": total_valid,
        "total_ignored_pixels": total_ignore,
        "land_pixels": total_land,
        "water_pixels": total_water,
        "pct_land": pct_land,
        "pct_water": pct_water,
        "pct_ignore": pct_ignore,
        "recommended_ce_weights": [round(w_land, 4), round(w_water, 4)],
        "recommended_bce_pos_weight": round(pos_weight, 2),
    }

    print("=" * 65)
    print("DATASET CLASS IMBALANCE AUDIT (Ground Truth Masks)")
    print("=" * 65)
    print(f"Masks Scanned      : {stats['num_files_scanned']}")
    print(f"Total Valid Pixels : {stats['total_valid_pixels']:,}")
    print(f"Cloud/Nodata Ignored: {stats['total_ignored_pixels']:,} ({stats['pct_ignore']:.2f}%)")
    print(f"Land Pixels (0)    : {stats['land_pixels']:,} ({stats['pct_land']:.2f}%)")
    print(f"Water Pixels (1)   : {stats['water_pixels']:,} ({stats['pct_water']:.2f}%)")
    print("-" * 65)
    print(f"Class Imbalance Ratio (Land : Water) : {stats['pct_land'] / max(stats['pct_water'], 0.001):.1f} : 1")
    print(f"Recommended CrossEntropy class weights [Land, Water]: {stats['recommended_ce_weights']}")
    print(f"Recommended BCEWithLogitsLoss pos_weight            : {stats['recommended_bce_pos_weight']}")
    print("=" * 65)

    return stats

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    lbl_dir = base_dir / "data" / "raw" / "sen1floods11" / "LabelHand"
    if not lbl_dir.exists():
        lbl_dir = base_dir / "data" / "raw"
    compute_class_imbalance(lbl_dir)
