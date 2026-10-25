import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# Robust raster reader fallback: tifffile or rasterio
try:
    import rasterio
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

import tifffile

def inspect_raster(path: str | Path, is_mask: bool = False) -> Dict[str, Any]:
    """
    Inspect geospatial raster attributes, bands, data type, nodata, 
    value ranges, and categorical distribution (if mask).
    Supports both tifffile and rasterio.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    meta: Dict[str, Any] = {
        "file_name": path.name,
    }

    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                meta["driver"] = src.driver
                meta["width"] = src.width
                meta["height"] = src.height
                meta["bands"] = src.count
                meta["dtypes"] = src.dtypes
                meta["crs"] = str(src.crs)
                meta["transform"] = list(src.transform)
                meta["nodata"] = src.nodata
                image = src.read()
        except Exception:
            # Fallback to tifffile
            image = tifffile.imread(path)
    else:
        image = tifffile.imread(path)

    # Ensure shape is (C, H, W)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    c, h, w = image.shape
    meta["array_shape"] = image.shape
    meta["bands"] = c
    meta["height"] = h
    meta["width"] = w
    meta["dtypes"] = str(image.dtype)
    meta["min"] = float(np.nanmin(image))
    meta["max"] = float(np.nanmax(image))
    meta["mean"] = float(np.nanmean(image))
    meta["has_nan"] = bool(np.isnan(image).any())

    print("=" * 60)
    print(f"FILE: {path.name}")
    print("=" * 60)
    print(f"Dimensions : {w} x {h} (Bands: {c})")
    print(f"Data type  : {meta['dtypes']}")
    print(f"Min / Max  : {meta['min']} / {meta['max']}")
    print(f"Mean       : {meta['mean']:.4f}")
    print(f"Has NaN    : {meta['has_nan']}")

    # Per band statistics
    if c > 1:
        print("\nPer-band Summary:")
        for b in range(c):
            band_data = image[b]
            print(f"  Band {b+1}: min={np.nanmin(band_data):.2f}, "
                  f"max={np.nanmax(band_data):.2f}, mean={np.nanmean(band_data):.2f}")

    # If it's a mask, calculate categorical frequency distribution
    if is_mask or "label" in path.name.lower() or "mask" in path.name.lower():
        unique, counts = np.unique(image, return_counts=True)
        distribution = dict(zip([int(x) for x in unique], [int(c) for c in counts]))
        meta["class_distribution"] = distribution
        total_px = image.size
        print("\nCategorical Class Distribution (Label Mask):")
        for val, cnt in distribution.items():
            pct = (cnt / total_px) * 100
            label_desc = "Unknown"
            if val == -1:
                label_desc = "Ignore / Nodata"
            elif val == 0:
                label_desc = "No water / Land"
            elif val == 1:
                label_desc = "Water / Flood"
            print(f"  Value {val:>2} ({label_desc}): {cnt:>8} px ({pct:>5.2f}%)")

    print("=" * 60)
    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a geospatial raster file.")
    parser.add_argument("path", type=str, help="Path to GeoTIFF file")
    parser.add_argument("--mask", action="store_true", help="Flag if this raster is a segmentation mask")
    args = parser.parse_args()
    inspect_raster(args.path, is_mask=args.mask)
