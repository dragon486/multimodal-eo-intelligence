"""
Geospatial Preprocessing Pipeline for Sen1Floods11 and Optical-SAR Pairs.
Handles band alignment, physical normalization, and mask sanitization.
Supports both tifffile and rasterio readers.
"""
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

import tifffile

# Standard S1 dB clipping range for VV and VH polarizations
S1_MIN_DB = -35.0
S1_MAX_DB = 10.0

# S2 reflectance scaling factor (standard Copernicus L1C/L2A to [0, 1])
S2_SCALE = 10000.0

# Prithvi HLS 6 bands index mapping from Sen1Floods11 13 bands:
# Sen1Floods11: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]
# Prithvi HLS : [B2, B3, B4, B8A, B11, B12] -> indices [1, 2, 3, 8, 11, 12]
PRITHVI_S2_INDICES = [1, 2, 3, 8, 11, 12]
RGB_INDICES = [3, 2, 1]  # [Red=B4, Green=B3, Blue=B2]

def read_tiff(path: Path) -> np.ndarray:
    """Read GeoTIFF raster array cleanly using rasterio or tifffile."""
    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                return src.read()
        except Exception:
            pass
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    return arr

def normalize_sar(s1_array: np.ndarray) -> np.ndarray:
    """
    Normalizes Sentinel-1 dual-pol SAR (VV, VH) in decibels.
    Clips extreme backscatter outliers [-35 dB, 10 dB] and maps to [0, 1].
    Shape: (2, H, W)
    """
    s1_clean = np.nan_to_num(s1_array, nan=S1_MIN_DB, neginf=S1_MIN_DB, posinf=S1_MAX_DB)
    s1_clipped = np.clip(s1_clean, S1_MIN_DB, S1_MAX_DB)
    s1_norm = (s1_clipped - S1_MIN_DB) / (S1_MAX_DB - S1_MIN_DB)
    return s1_norm.astype(np.float32)

def normalize_optical(s2_array: np.ndarray, select_prithvi_bands: bool = False) -> np.ndarray:
    """
    Normalizes Sentinel-2 multispectral surface reflectance.
    Scales DN [0, 10000] -> [0.0, 1.0].
    Optionally extracts the 6 HLS-compatible bands for Prithvi foundation models.
    Shape: (C, H, W)
    """
    if select_prithvi_bands:
        s2_array = s2_array[PRITHVI_S2_INDICES, :, :]

    s2_clean = np.nan_to_num(s2_array, nan=0.0, neginf=0.0)
    s2_norm = np.clip(s2_clean / S2_SCALE, 0.0, 1.0)
    return s2_norm.astype(np.float32)

def sanitize_mask(mask_array: np.ndarray) -> np.ndarray:
    """
    Sanitizes ground truth flood segmentation masks.
    Sen1Floods11 convention:
      -1 : Nodata / cloud / unlabelled (for loss ignore_index)
       0 : Non-water (background)
       1 : Water (flood / permanent water)
    """
    if mask_array.ndim == 3:
        mask_array = mask_array[0]
    return mask_array.astype(np.int64)

def load_and_preprocess_pair(
    s1_path: Path,
    s2_path: Path,
    label_path: Optional[Path] = None,
    prithvi_bands: bool = False
) -> Dict[str, Any]:
    """
    Loads raw GeoTIFFs, applies physical calibration and normalization,
    and returns sanitized numpy arrays.
    """
    s1_raw = read_tiff(Path(s1_path))
    s2_raw = read_tiff(Path(s2_path))

    s1_processed = normalize_sar(s1_raw)
    s2_processed = normalize_optical(s2_raw, select_prithvi_bands=prithvi_bands)

    sample = {
        "s1": s1_processed,        # (2, H, W)
        "s2": s2_processed,        # (13 or 6, H, W)
        "s1_path": str(s1_path),
        "s2_path": str(s2_path),
    }

    if label_path and Path(label_path).exists():
        label_raw = read_tiff(Path(label_path))
        sample["mask"] = sanitize_mask(label_raw)  # (H, W)
        sample["label_path"] = str(label_path)

    return sample
