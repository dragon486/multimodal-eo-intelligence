# Shared Data Contract: Person 1 (Model Lead) & Person 2 (Data & Evaluation Lead)

**Project:** Foundation Vision Models for Multimodal Remote Sensing and Environmental Intelligence  
**Task Focus:** Multimodal Flood Segmentation & Land-Cover Mapping (Sen1Floods11 & BigEarthNet)  
**Document Version:** 1.1.0  
**Status:** Approved & Verified  

---

## 1. Input Tensor Specifications

When Person 1 initializes the PyTorch DataLoader via `from src.data.datasets import get_dataloader`, each batch dictionary contains the following keys:

| Key | Tensor Shape | Data Type | Value Range | Channel Order / Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `"s1"` | `(B, 2, 512, 512)` | `torch.float32` | `[0.0, 1.0]` | Channel 0: VV polarization (dB normalized)<br>Channel 1: VH polarization (dB normalized) |
| `"s2"` (Default) | `(B, 13, 512, 512)` | `torch.float32` | `[0.0, 1.0]` | Full 13 Sentinel-2 MSI bands (`B1` through `B12`, `B8A`) scaled by `10000` |
| `"s2"` (Prithvi Mode) | `(B, 6, 512, 512)` | `torch.float32` | `[0.0, 1.0]` | Extracted 6 HLS bands: `[B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR), B11 (SWIR 1), B12 (SWIR 2)]` |
| `"mask"` | `(B, 512, 512)` | `torch.long` | `{-1, 0, 1}` | Categorical Ground Truth segmentation mask |
| `"chip_id"` | List of strings | `str` | N/A | Unique geospatial chip ID (e.g. `Ghana_103272`) |

---

## 2. Ground Truth Semantic Encoding

For Sen1Floods11 flood water mapping:

- **`-1` : Ignore / Invalid / Cloud Obscured**  
  *Must be ignored during gradient descent.* In PyTorch, configure your loss with `ignore_index=-1`:
  ```python
  criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
  ```
- **`0` : Non-water / Land Background** (Dominant class: ~89.83%)
- **`1` : Water / Flood Extent** (Target class: ~10.17%)

---

## 3. Class Imbalance Mitigation

Empirically derived loss weights via `src/data/class_balance.py`:
```python
weights = torch.tensor([0.5566, 4.9183], dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
```
