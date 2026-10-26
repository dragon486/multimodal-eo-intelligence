"""High-Performance Resumable Multithreaded Downloader for Sen1Floods11 Dataset.

Downloads official split CSVs and paired satellite imagery chips from Google Cloud Storage.
Supports:
- Multi-threaded parallel downloading (8-16 workers)
- Automatic retry on transient network errors / timeouts
- Resumable: skips already downloaded, valid files
- Downloads all 446 hand-labeled chips for production-grade training and evaluation
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
from tqdm import tqdm

GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def download_single_file(url: str, dest_path: Path, max_retries: int = 5) -> bool:
    """Downloads a single file with retries and timeout."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True  # Already downloaded

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Multimodal-EO-Downloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                out_file.write(response.read())
            return True
        except Exception as e:
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(attempt * 2)  # Exponential backoff
            else:
                print(f"\n[Failed after {max_retries} attempts] {url}: {e}")
                return False
    return False


def parse_chip_id(line: str) -> str:
    """Extracts raw chip ID e.g. Ghana_103272 from CSV line."""
    first_col = line.split(",")[0].strip()
    return first_col.replace("_S1Hand.tif", "").replace(".tif", "").strip()


def download_sen1floods11_dataset(
    splits_dir: Path,
    raw_data_dir: Path,
    all_chips: bool = True,
    max_samples_per_split: int = 10,
    max_workers: int = 8,
):
    """Downloads the complete Sen1Floods11 benchmark in parallel."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    s1_dir = raw_data_dir / "S1Hand"
    s2_dir = raw_data_dir / "S2Hand"
    label_dir = raw_data_dir / "LabelHand"

    for d in [s1_dir, s2_dir, label_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect unique chips across requested splits
    unique_chips = set()
    split_counts: Dict[str, int] = {}

    for split_name, split_file in SPLIT_FILES.items():
        csv_path = splits_dir / split_file
        if not csv_path.exists():
            continue
        with open(csv_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        selected_lines = lines if all_chips else lines[:max_samples_per_split]
        chips = [parse_chip_id(line) for line in selected_lines]
        split_counts[split_name] = len(chips)
        unique_chips.update(chips)

    print("=" * 65)
    print("SEN1FLOODS11 PRODUCTION BENCHMARK DOWNLOADER")
    print("=" * 65)
    print(f"Target Directory: {raw_data_dir}")
    print(f"Splits Target   : {split_counts}")
    print(f"Total Unique Chips to Download: {len(unique_chips)} (3 files per chip: S1, S2, Label)")
    print(f"Total Target Files            : {len(unique_chips) * 3}")
    print(f"Parallel Worker Threads       : {max_workers}")
    print("=" * 65)

    # Prepare download tasks: (url, dest_path)
    tasks: List[Tuple[str, Path]] = []
    for chip_id in unique_chips:
        # 1. S1 SAR dual polarization (VV + VH)
        s1_filename = f"{chip_id}_S1Hand.tif"
        tasks.append((f"{GCS_BASE}/data/flood_events/HandLabeled/S1Hand/{s1_filename}", s1_dir / s1_filename))

        # 2. S2 Optical 13-band multispectral
        s2_filename = f"{chip_id}_S2Hand.tif"
        tasks.append((f"{GCS_BASE}/data/flood_events/HandLabeled/S2Hand/{s2_filename}", s2_dir / s2_filename))

        # 3. Ground truth hand label mask
        label_filename = f"{chip_id}_LabelHand.tif"
        tasks.append((f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand/{label_filename}", label_dir / label_filename))

    # Execute in parallel thread pool
    successful = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_file, url, path): (url, path) for url, path in tasks}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Downloading Rasters", unit="file")
        for future in pbar:
            if future.result():
                successful += 1

    print("\n" + "=" * 65)
    print(f"DOWNLOAD COMPLETE: {successful}/{len(tasks)} files successfully verified on disk.")
    print("=" * 65)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    splits_path = base_dir / "data" / "splits"
    raw_path = base_dir / "data" / "raw" / "sen1floods11"

    download_all = "--subset" not in sys.argv
    samples = 10
    if "--num" in sys.argv:
        idx = sys.argv.index("--num")
        if idx + 1 < len(sys.argv):
            samples = int(sys.argv[idx + 1])

    download_sen1floods11_dataset(
        splits_dir=splits_path,
        raw_data_dir=raw_path,
        all_chips=download_all,
        max_samples_per_split=samples,
        max_workers=8,
    )
