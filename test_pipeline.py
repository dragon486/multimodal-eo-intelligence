import sys
from pathlib import Path
print("Step 1: Imports starting...", flush=True)

from src.data.preprocess import load_and_preprocess_pair
print("Step 2: Preprocess loaded.", flush=True)

from src.evaluation.metrics import compute_segmentation_metrics
print("Step 3: Metrics loaded.", flush=True)

from src.evaluation.visualize import plot_multimodal_sample
print("Step 4: Visualize loaded.", flush=True)

s1_p = Path("data/raw/test_s1.tif")
s2_p = Path("data/raw/test_s2.tif")
lbl_p = Path("data/raw/test_label.tif")

print("Step 5: Preprocessing sample pair...", flush=True)
data = load_and_preprocess_pair(s1_p, s2_p, lbl_p)
print(f"  S1 normalized shape: {data['s1'].shape}, min={data['s1'].min():.2f}, max={data['s1'].max():.2f}")
print(f"  S2 normalized shape: {data['s2'].shape}, min={data['s2'].min():.2f}, max={data['s2'].max():.2f}")
print(f"  Mask shape: {data['mask'].shape}")

print("Step 6: Computing metrics on Ground Truth...", flush=True)
m = compute_segmentation_metrics(data["mask"], data["mask"], ignore_index=-1)
print(f"  Mean IoU: {m['mean_iou']:.4f}")
print(f"  Water IoU: {m['water_iou']:.4f}")
print(f"  Accuracy: {m['accuracy']:.4f}")
print(f"  Ignored px (cloud/nodata): {m['ignored_pixels']}")

print("Step 7: Generating test visualization plot...", flush=True)
out_fig = Path("reports/visualizations/ghana_sample_verification.png")
plot_multimodal_sample(
    s1=data["s1"],
    s2=data["s2"],
    target=data["mask"],
    pred=data["mask"],
    chip_id="Ghana_103272_Test",
    metrics=m,
    save_path=out_fig
)
print(f"Step 8: Successfully generated visualization at: {out_fig}", flush=True)
print("ALL TESTS PASSED!", flush=True)
