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
- **`0` : Non-water / Land Background** (Dominant class: ~98.3%)
- **`1` : Water / Flood Extent** (Positive target class: ~0.59% to 2.5%)

---

## 3. Critical Prioritized Checks for Person 1

### Check 1: Severe Class Imbalance Mitigation
- **The Problem:** Flood water pixels represent only **~0.59%** of valid pixels in typical scenes. An unweighted model can predict class `0` everywhere and achieve **98.3% accuracy** while having **0% Water IoU**.
- **Action for Person 1:**
  1. **Do NOT rank models by overall pixel accuracy alone.** Always prioritize **`water_iou`** and **`water_f1`**.
  2. Use **Weighted Cross-Entropy** or **Compound Loss (Dice + CE)**:
     ```python
     # Option A: Weighted CrossEntropy with ignore_index
     # Weights derived empirically via src/data/class_balance.py:
     class_weights = torch.tensor([1.0, 50.0], dtype=torch.float32).to(device)
     criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
     ```
     ```python
     # Option B: Compound Dice + Weighted CE Loss
     loss = dice_loss(pred, target, ignore_index=-1) + 0.5 * ce_loss(pred, target)
     ```

### Check 2: Split Integrity & Held-out Event Guarantee
- **Disjoint Splits:** Verified 0 chip overlap between `train`, `val`, and `test`.
- **Geographic Generalization Test:** The **Bolivia flood event is 100% excluded** from `flood_train_data.csv` (0 chips) and `flood_valid_data.csv` (0 chips). It is strictly reserved in `flood_bolivia_data.csv` (15 chips) to test whether your foundation model generalizes to unseen geographical terrains.

### Check 3: Preprocessing Consistency Guarantee
- The training data loader (`src/data/datasets.py`), the inspection tools (`src/data/inspect_raster.py`), and the interactive dashboard (`src/app/dashboard.py`) all import from the **single source of truth**: `src/data/preprocess.py`.
- SAR decibel clipping range: `[-35.0 dB, +10.0 dB]` normalized to `[0.0, 1.0]`.
- Optical surface reflectance: `DN / 10000.0` clipped to `[0.0, 1.0]`.
- Identical band order is guaranteed across all phases.

---

## 4. Normalization Formula Applied

- **Sentinel-1 SAR:**
  $$\text{dB} = \text{clip}(\text{dB}, -35.0, 10.0)$$
  $$\text{S1}_{\text{norm}} = \frac{\text{dB} - (-35.0)}{10.0 - (-35.0)} \in [0.0, 1.0]$$

- **Sentinel-2 Optical:**
  $$\text{S2}_{\text{norm}} = \text{clip}\left(\frac{\text{DN}}{10000.0}, 0.0, 1.0\right)$$

---

## 5. Evaluation Function Interface

Person 1 can evaluate any model checkpoint against ground truth using the standardized metric functions implemented by Person 2:

```python
from src.evaluation.metrics import compute_segmentation_metrics, MetricAccumulator

# Batch evaluation
metrics = compute_segmentation_metrics(
    pred=pred_tensor,       # (B, H, W) or (H, W) argmax predictions
    target=target_tensor,   # (B, H, W) or (H, W) labels
    ignore_index=-1,
    num_classes=2
)
# Returns: dict with 'mean_iou', 'water_iou', 'land_iou', 'water_f1', 'accuracy', etc.
```

For full test-epoch evaluation:
```python
accumulator = MetricAccumulator(num_classes=2, ignore_index=-1)
for batch in test_loader:
    preds = model(batch["s1"], batch["s2"]).argmax(dim=1)
    accumulator.update(preds, batch["mask"])

epoch_results = accumulator.compute()
print(f"Validation Water IoU: {epoch_results['water_iou']:.4f}")
```
