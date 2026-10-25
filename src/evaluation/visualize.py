"""
Geospatial visualization utilities.
Creates 3-panel and 4-panel comparison plots:
- Sentinel-1 SAR (VV/VH false color or backscatter)
- Sentinel-2 Optical (RGB true color or False Color NIR/SWIR)
- Ground Truth Mask
- Model Prediction Mask
"""
from pathlib import Path
from typing import Optional, Any, Dict
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Standard Colormap for Sen1Floods11
# -1: Grey (Ignore / Nodata)
#  0: Greenish / Khaki (Land / Non-water)
#  1: Deep Blue (Water / Flood)
MASK_COLORS = ["#555555", "#90a955", "#1d3557"]
MASK_CMAP = mcolors.ListedColormap(MASK_COLORS)
MASK_BOUNDS = [-1.5, -0.5, 0.5, 1.5]
MASK_NORM = mcolors.BoundaryNorm(MASK_BOUNDS, MASK_CMAP.N)

def to_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def make_s1_rgb(s1: np.ndarray) -> np.ndarray:
    """
    Creates a 3-channel false-color RGB image from dual-pol SAR (VV, VH).
    R: VV, G: VH, B: VV / (VH + 1e-6)
    Expects normalized inputs of shape (2, H, W).
    """
    vv = s1[0]
    vh = s1[1]
    ratio = np.clip(vv / (vh + 1e-3), 0.0, 1.0)
    rgb = np.stack([vv, vh, ratio], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def make_s2_rgb(s2: np.ndarray, is_prithvi_bands: bool = False) -> np.ndarray:
    """
    Extracts RGB channels for visualization.
    If 13-band S2: indices [3, 2, 1] correspond to Red (B4), Green (B3), Blue (B2).
    If 6-band Prithvi: indices [2, 1, 0] correspond to Red (B4), Green (B3), Blue (B2).
    """
    if is_prithvi_bands:
        rgb = np.stack([s2[2], s2[1], s2[0]], axis=-1)
    elif s2.shape[0] >= 4:
        rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1)
    else:
        rgb = np.repeat(s2[0:1], 3, axis=0).transpose(1, 2, 0)
    
    # Percentile stretch for visualization contrast
    p2, p98 = np.percentile(rgb, (2, 98))
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    return rgb

def plot_multimodal_sample(
    s1: Any,
    s2: Any,
    target: Any,
    pred: Optional[Any] = None,
    chip_id: str = "Sample",
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str | Path] = None
):
    """
    Plots a multi-panel comparison figure.
    Panels:
      1. Sentinel-1 SAR Dual-Pol Composite
      2. Sentinel-2 Optical RGB Composite
      3. Ground Truth Flood Mask
      4. (Optional) Model Prediction Mask
    """
    s1 = to_numpy(s1)
    s2 = to_numpy(s2)
    target = to_numpy(target)
    if pred is not None:
        pred = to_numpy(pred)

    num_panels = 4 if pred is not None else 3
    fig, axes = plt.subplots(1, num_panels, figsize=(4.5 * num_panels, 4.5), dpi=150)

    # 1. S1 SAR
    s1_display = make_s1_rgb(s1)
    axes[0].imshow(s1_display)
    axes[0].set_title("Sentinel-1 SAR\n(VV/VH Composite)", fontsize=10)
    axes[0].axis("off")

    # 2. S2 Optical
    s2_display = make_s2_rgb(s2)
    axes[1].imshow(s2_display)
    axes[1].set_title("Sentinel-2 Optical\n(RGB Composite)", fontsize=10)
    axes[1].axis("off")

    # 3. Ground Truth
    axes[2].imshow(target, cmap=MASK_CMAP, norm=MASK_NORM)
    axes[2].set_title("Ground Truth Label\n(Grey=Ignore, Green=Land, Blue=Water)", fontsize=10)
    axes[2].axis("off")

    # 4. Model Prediction (if provided)
    if pred is not None:
        axes[3].imshow(pred, cmap=MASK_CMAP, norm=MASK_NORM)
        title_str = "Model Prediction"
        if metrics:
            title_str += f"\n(mIoU: {metrics.get('mean_iou', 0):.2f} | Water IoU: {metrics.get('water_iou', 0):.2f})"
        axes[3].set_title(title_str, fontsize=10)
        axes[3].axis("off")

    plt.suptitle(f"Chip: {chip_id}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
