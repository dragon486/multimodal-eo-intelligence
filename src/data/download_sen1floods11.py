"""
Downloader utility for Sen1Floods11 dataset.
Downloads official split CSVs and imagery chips from Google Cloud Storage public buckets.
"""
import os
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm

GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv"
}

def download_file(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    try:
        urllib.request.urlretrieve(url, dest_path)
        return True
    except Exception as e:
        print(f"  [Error] {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_splits(dest_dir: Path):
    print(">>> Downloading Sen1Floods11 official split CSVs from GCS...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in SPLIT_FILES.items():
        url = f"{GCS_BASE}/splits/flood_handlabeled/{filename}"
        out_file = dest_dir / filename
        if download_file(url, out_file):
            print(f"  [OK] Saved {filename}")
        else:
            print(f"  [FAILED] Could not download {filename}")

def parse_chip_id(line: str) -> str:
    """Extracts raw chip ID e.g. Ghana_103272 from CSV line."""
    first_col = line.split(",")[0].strip()
    return first_col.replace("_S1Hand.tif", "").replace(".tif", "").strip()

def download_subset(splits_dir: Path, raw_data_dir: Path, max_samples_per_split: int = 10):
    """
    Downloads a lightweight initial subset (e.g. 10 samples per split)
    so Person 2 and Person 1 can immediately run inspection, preprocessing,
    and baseline model debugging.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    s1_dir = raw_data_dir / "S1Hand"
    s2_dir = raw_data_dir / "S2Hand"
    label_dir = raw_data_dir / "LabelHand"

    for d in [s1_dir, s2_dir, label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for split_name, split_file in SPLIT_FILES.items():
        csv_path = splits_dir / split_file
        if not csv_path.exists():
            continue
        
        with open(csv_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        chips = [parse_chip_id(line) for line in lines[:max_samples_per_split]]
        print(f"\n>>> Downloading {len(chips)} chips for '{split_name}' split...")

        for chip_id in tqdm(chips, desc=f"Split: {split_name}"):
            # S1 SAR dual polarization (VV + VH)
            s1_filename = f"{chip_id}_S1Hand.tif"
            s1_url = f"{GCS_BASE}/data/flood_events/HandLabeled/S1Hand/{s1_filename}"
            download_file(s1_url, s1_dir / s1_filename)

            # S2 Optical 13-band multispectral
            s2_filename = f"{chip_id}_S2Hand.tif"
            s2_url = f"{GCS_BASE}/data/flood_events/HandLabeled/S2Hand/{s2_filename}"
            download_file(s2_url, s2_dir / s2_filename)

            # Quality-controlled hand label mask
            label_filename = f"{chip_id}_LabelHand.tif"
            label_url = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand/{label_filename}"
            download_file(label_url, label_dir / label_filename)

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    splits_path = base_dir / "data" / "splits"
    raw_path = base_dir / "data" / "raw" / "sen1floods11"

    download_splits(splits_path)
    
    if "--subset" in sys.argv:
        n = 5
        if "--num" in sys.argv:
            idx = sys.argv.index("--num")
            if idx + 1 < len(sys.argv):
                n = int(sys.argv[idx + 1])
        download_subset(splits_path, raw_path, max_samples_per_split=n)
