# Dataset & Geospatial Pipeline Report (Person 2)

**Author:** Person 2 (Geospatial Data & Environmental Intelligence Lead)  
**Project:** Foundation Vision Models for Multimodal Remote Sensing & Environmental Intelligence  
**Focus Dataset:** Sen1Floods11 (v1.1 Hand-Labeled Optical & SAR Benchmark)  
**Status:** 60-Chip Subset Downloaded & Fully Audited  

---

## 1. Verified Official Splits & Leakage Audit

The official dataset split files were downloaded from `gs://sen1floods11/v1.1/splits/flood_handlabeled/` and audited via [`src/data/split.py`](file:///c:/Users/adel/Downloads/Projects/fv_model/src/data/split.py):

| Split | Total Catalog Chips | Local Subset Chips | Regional Representation | Leakage Status |
| :--- | :---: | :---: | :--- | :--- |
| **Train** | 252 | 15 | Ghana, India, Mekong, USA, Nigeria | **0 chip overlap** |
| **Validation** | 89 | 15 | Ghana, India, Mekong, USA, Pakistan | **0 chip overlap** |
| **Test** | 90 | 15 | Ghana, India, Mekong, USA, Spain | **0 chip overlap** |
| **Bolivia (Held-out)** | 15 | 15 (100%) | Bolivia flood event only | **100% Isolated from Train/Val** |

---

## 2. Empirical Class Imbalance & Loss Weighting Audit

Executed across 50 hand-labeled masks (over **13.1 million pixels** analyzed via [`src/data/class_balance.py`](file:///c:/Users/adel/Downloads/Projects/fv_model/src/data/class_balance.py)):

- **Total Pixels Evaluated:** 13,107,200
- **Cloud / Invalid Obscuration (`-1`):** 3,591,959 pixels (**27.40%**) — ignored during loss computation (`ignore_index=-1`).
- **Valid Land Pixels (`0`):** 8,547,915 pixels (**89.83%**)
- **Valid Flood Water Pixels (`1`):** 967,326 pixels (**10.17%**)
- **Class Imbalance Ratio (Land : Water):** **8.8 : 1**

### Recommended Loss Configuration for Person 1:
```python
# CrossEntropy with empirical class weights:
weights = torch.tensor([0.5566, 4.9183], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

# Or BCEWithLogitsLoss:
pos_weight = torch.tensor([8.84]).to(device)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

## 3. Radiometric & Sensor Physics Specifications

1. **Sentinel-1 SAR:**
   - C-Band Dual Polarization: Channel 0 = `VV`, Channel 1 = `VH`.
   - Raw unit: Radar backscatter $\sigma^0$ (dB).
   - Normalization: Clipped to $[-35.0\text{ dB}, +10.0\text{ dB}]$ and linearly scaled to $[0.0, 1.0]$.
2. **Sentinel-2 Optical (MSI):**
   - 13 multispectral bands (10m, 20m, 60m resampled to 10m GSD).
   - Normalization: Scaled by $10,000$ (reflectance $[0.0, 1.0]$).
   - Prithvi foundation model mode: extracts 6 HLS bands `[B2, B3, B4, B8A, B11, B12]`.

---

## 4. Generated Verification Artifacts

All rendered 4-panel comparison figures are stored in [`reports/visualizations/`](file:///c:/Users/adel/Downloads/Projects/fv_model/reports/visualizations/):
- [`scene_Bolivia_103757.png`](file:///c:/Users/adel/Downloads/Projects/fv_model/reports/visualizations/scene_Bolivia_103757.png)
- [`scene_Bolivia_129334.png`](file:///c:/Users/adel/Downloads/Projects/fv_model/reports/visualizations/scene_Bolivia_129334.png)
- [`scene_Ghana_103272.png`](file:///c:/Users/adel/Downloads/Projects/fv_model/reports/visualizations/scene_Ghana_103272.png)
- [`scene_Ghana_134751.png`](file:///c:/Users/adel/Downloads/Projects/fv_model/reports/visualizations/scene_Ghana_134751.png)
- [`scene_Ghana_142312.png`](file:///c:/Users/adel/Downloads/Projects/fv_model/reports/visualizations/scene_Ghana_142312.png)

---

## 5. Live Environmental Intelligence Dashboard

- Launch command: `streamlit run src/app/dashboard.py`
- Active URL: `http://localhost:8501`
- Features: Real-time chip browsing, multi-sensor comparison, flooded area estimation (hectares and km²), NDWI index, and automated report generation.
