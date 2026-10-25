"""
Batch visualization generator for multiple geographical scenes.
Produces publication-quality 4-panel figures for reports and team presentation.
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocess import load_and_preprocess_pair
from src.evaluation.metrics import compute_segmentation_metrics
from src.evaluation.visualize import plot_multimodal_sample

def generate_scene_visualizations(max_scenes: int = 5):
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = base_dir / "data" / "raw" / "sen1floods11"
    out_dir = base_dir / "reports" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    lbl_dir = raw_dir / "LabelHand"
    s1_dir = raw_dir / "S1Hand"
    s2_dir = raw_dir / "S2Hand"

    if not lbl_dir.exists():
        print(f"[!] {lbl_dir} does not exist yet.")
        return

    label_files = list(lbl_dir.glob("*_LabelHand.tif"))[:max_scenes]
    print(f">>> Generating 4-panel visualizations for {len(label_files)} scenes...")

    for lbl_path in label_files:
        chip_id = lbl_path.name.replace("_LabelHand.tif", "")
        s1_path = s1_dir / f"{chip_id}_S1Hand.tif"
        s2_path = s2_dir / f"{chip_id}_S2Hand.tif"

        if not (s1_path.exists() and s2_path.exists()):
            continue

        data = load_and_preprocess_pair(s1_path, s2_path, lbl_path)
        metrics = compute_segmentation_metrics(data["mask"], data["mask"], ignore_index=-1)

        out_img = out_dir / f"scene_{chip_id}.png"
        plot_multimodal_sample(
            s1=data["s1"],
            s2=data["s2"],
            target=data["mask"],
            pred=data["mask"],
            chip_id=chip_id,
            metrics=metrics,
            save_path=out_img
        )
        print(f"  [OK] Rendered {chip_id} -> {out_img.name}")

if __name__ == "__main__":
    generate_scene_visualizations(max_scenes=6)
