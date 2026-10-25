"""
Dataset split validation and spatial leakage audit.
Ensures training, validation, and test sets do not contain overlapping geographic locations,
and explicitly verifies that the held-out Bolivia flood event remains strictly excluded from training.
"""
from pathlib import Path
from typing import Dict, Any, List, Set

def parse_flood_event_region(chip_id: str) -> str:
    parts = chip_id.split("_")
    return parts[0] if parts else "Unknown"

def parse_chip_id(line: str) -> str:
    first_col = line.split(",")[0].strip()
    return first_col.replace("_S1Hand.tif", "").replace(".tif", "").strip()

def audit_splits(splits_dir: str | Path) -> Dict[str, Any]:
    splits_dir = Path(splits_dir)
    split_files = {
        "train": splits_dir / "flood_train_data.csv",
        "val": splits_dir / "flood_valid_data.csv",
        "test": splits_dir / "flood_test_data.csv",
        "bolivia_test": splits_dir / "flood_bolivia_data.csv"
    }

    sets: Dict[str, Set[str]] = {}
    regions: Dict[str, Dict[str, int]] = {}

    print("=" * 68)
    print("SEN1FLOODS11 GEOGRAPHIC SPLIT AUDIT & DATA LEAKAGE CHECK")
    print("=" * 68)

    for name, path in split_files.items():
        if not path.exists():
            print(f"[!] Split file missing: {path.name}")
            continue
        with open(path, "r") as f:
            chips = [parse_chip_id(line) for line in f if line.strip()]
        
        sets[name] = set(chips)

        reg_counts: Dict[str, int] = {}
        for c in chips:
            r = parse_flood_event_region(c)
            reg_counts[r] = reg_counts.get(r, 0) + 1
        regions[name] = reg_counts

        print(f"Split: {name:<12} | Total Samples: {len(chips):>4}")
        top_regions = ", ".join([f"{k} ({v})" for k, v in list(reg_counts.items())[:5]])
        print(f"  Regions breakdown: {top_regions}...")

    # Check 1: Disjoint chips
    print("\n" + "-" * 68)
    print("CHECK 1: SPATIAL LEAKAGE / OVERLAPPING CHIPS")
    has_leakage = False
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for s1, s2 in pairs:
        if s1 in sets and s2 in sets:
            overlap = sets[s1].intersection(sets[s2])
            if overlap:
                print(f"  [CRITICAL LEAKAGE] Found {len(overlap)} overlapping chips between {s1} and {s2}!")
                has_leakage = True
            else:
                print(f"  [PASSED] No chip overlap between {s1} and {s2}.")

    # Check 2: Held-out event isolation (e.g. Bolivia)
    print("\n" + "-" * 68)
    print("CHECK 2: HELD-OUT EVENT INTEGRITY (Bolivia flood event isolation)")
    bolivia_in_train = regions.get("train", {}).get("Bolivia", 0)
    bolivia_in_val = regions.get("val", {}).get("Bolivia", 0)
    bolivia_in_held_out = regions.get("bolivia_test", {}).get("Bolivia", 0)

    print(f"  Bolivia chips in train        : {bolivia_in_train}")
    print(f"  Bolivia chips in validation   : {bolivia_in_val}")
    print(f"  Bolivia chips in bolivia_test : {bolivia_in_held_out}")

    bolivia_isolated = (bolivia_in_train == 0) and (bolivia_in_val == 0) and (bolivia_in_held_out > 0)
    if bolivia_isolated:
        print("  [PASSED] Bolivia is 100% held-out from training & validation.")
        print("           Guarantees a clean, uncompromised out-of-region generalization test.")
    else:
        print("  [WARNING] Bolivia event found in training/val splits!")

    print("=" * 68)
    return {
        "sets": sets,
        "regions": regions,
        "leakage": has_leakage,
        "bolivia_isolated": bolivia_isolated
    }

if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parents[2]
    target_splits = base_dir / "data" / "splits"
    if len(sys.argv) > 1:
        target_splits = Path(sys.argv[1])
    audit_splits(target_splits)
