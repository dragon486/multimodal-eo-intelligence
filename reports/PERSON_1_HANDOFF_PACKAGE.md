# 🛰️ Multimodal Remote Sensing Foundation Models: Technical Handoff Package for Person 1

**From:** Person 2 (Geospatial Data & Environmental Intelligence Lead)  
**To:** Person 1 (Foundation Models & Multimodal Learning Lead)  
**Project:** Foundation Vision Models for Multimodal Remote Sensing and Environmental Intelligence  
**Document Version:** 1.0.0 (Production Verified)  
**Workspace:** `c:\Users\adel\Downloads\Projects\fv_model`  

---

## 📌 1. Your Mission & Role as Person 1

You are the **Model Development Lead**. Your responsibility is to design, implement, and train the neural architectures:
1. **Foundation Model Backbone:** Adapt a pretrained Earth-observation foundation model (primary baseline: **Prithvi-EO-2.0 100M-TL**; alternative: **Clay** / **TerraMind**).
2. **Multimodal Fusion Engine:** Build the fusion module that combines **Sentinel-1 SAR (Radar)** and **Sentinel-2 Optical (Multispectral)** representations.
3. **Task-Specific Heads & Training:** Train flood segmentation decoders (and land-cover heads) using parameter-efficient fine-tuning or frozen backbones.
4. **Model Delivery:** Provide trained model checkpoints and an inference function that Person 2 can plug directly into the live Environmental Dashboard.

---

## 📋 2. The Shared Data Contract (Data Format & Loader)

Person 2 has already downloaded, calibrated, and preprocessed the benchmark dataset (**Sen1Floods11 v1.1**) into `data/raw/sen1floods11/`. You do not need to write raw GeoTIFF readers or handle geospatial coordinate transformations.

### Ready-to-Use PyTorch DataLoader
You can import the standardized DataLoader directly in your training scripts:

```python
from src.data.datasets import get_dataloader

# Standard mode: 13-band Sentinel-2 + 2-channel SAR
train_loader = get_dataloader(
    split_csv="data/splits/flood_train_data.csv",
    raw_data_dir="data/raw/sen1floods11",
    batch_size=4,
    shuffle=True,
    prithvi_bands=False  # Set True for 6-band Prithvi mode
)

val_loader = get_dataloader(
    split_csv="data/splits/flood_valid_data.csv",
    raw_data_dir="data/raw/sen1floods11",
    batch_size=4,
    shuffle=False,
    prithvi_bands=False
)
```

### Batch Dictionary Specification

Each iteration yielding `batch = next(iter(train_loader))` provides:

| Key | Tensor Shape | Type | Value Range | Channel Semantics |
| :--- | :--- | :--- | :--- | :--- |
| `batch["s1"]` | `(B, 2, 512, 512)` | `torch.float32` | `[0.0, 1.0]` | **SAR Dual-Pol:**<br>Ch 0: `VV` backscatter (clipped `[-35, 10]` dB)<br>Ch 1: `VH` backscatter (clipped `[-35, 10]` dB) |
| `batch["s2"]` *(Default)* | `(B, 13, 512, 512)` | `torch.float32` | `[0.0, 1.0]` | **Sentinel-2 MSI:**<br>All 13 spectral bands (`B1`–`B12`, `B8A`) scaled by `10000.0` |
| `batch["s2"]` *(Prithvi Mode)* | `(B, 6, 512, 512)` | `torch.float32` | `[0.0, 1.0]` | **HLS 6 Bands for Prithvi-EO-2.0:**<br>`[B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR), B11 (SWIR 1), B12 (SWIR 2)]` |
| `batch["mask"]` | `(B, 512, 512)` | `torch.long` | `{-1, 0, 1}` | **Ground Truth Flood Mask:**<br>`-1` = Ignore (cloud / nodata)<br>`0` = Non-water (Land)<br>`1` = Flood / Open Water |
| `batch["chip_id"]` | `List[str]` | `str` | N/A | Geospatial scene identifier (e.g., `Ghana_103272`, `Bolivia_103757`) |

---

## ⚠️ 3. Three Critical Rules to Avoid Failing Experiments

### Rule 1: Mitigate Severe Class Imbalance (Do NOT Use Unweighted Loss)
We evaluated over **13.1 million pixels** across the ground-truth masks:
* **Land background (`0`):** `89.83%` of valid pixels
* **Flood water (`1`):** `10.17%` of valid pixels (in some scenes as low as `0.59%`)
* **Cloud-obscured ignored pixels (`-1`):** `27.40%` of all raw pixels

> **Danger:** If you use unweighted CrossEntropyLoss, your model will predict class `0` everywhere, report **90%+ overall accuracy**, but have **0% Water IoU**.

**Your Loss Configuration:**
```python
import torch
import torch.nn as nn

# Recommended empirical class weights: Land=0.5566, Water=4.9183
weights = torch.tensor([0.5566, 4.9183], dtype=torch.float32).to(device)

# Crucial: ignore_index=-1 ensures clouds/nodata do not produce gradients
criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
```

Alternatively, use a **Compound Loss (Dice Loss + Weighted CrossEntropy)**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Dice}}(\text{pred}, \text{target}) + 0.5 \cdot \mathcal{L}_{\text{Weighted-CE}}(\text{pred}, \text{target})$$

---

### Rule 2: Respect Split Integrity & Held-Out Event Isolation
* `flood_train_data.csv`: 252 catalog chips (15 in local test subset)
* `flood_valid_data.csv`: 89 catalog chips (15 in local test subset)
* `flood_test_data.csv`: 90 catalog chips (15 in local test subset)
* `flood_bolivia_data.csv`: 15 chips (**100% Held-Out Generalization Set**)

> **Do NOT touch the Bolivia split during training or hyperparameter tuning.** It is strictly reserved for evaluating out-of-distribution geographic transfer.

---

### Rule 3: Benchmark Metric Hierarchy
When compiling your experimental tables, report the metrics produced by Person 2's evaluation module:
1. **Primary Metric:** `water_iou` (Intersection over Union for class 1)
2. **Secondary Metric:** `water_f1` and `macro_f1`
3. **Tertiary Metric:** `mean_iou` (mIoU)
4. *Do not report pixel accuracy as your main metric.*

---

## 🏗️ 4. Model Architecture & Multimodal Fusion Guide

### Experiment Matrix You Must Execute

```
                  ┌──────────────────────┐
                  │  Sentinel-2 Optical  │
                  └──────────┬───────────┘
                             │
                     [Optical Encoder]
                             │
                             ▼ Z_optical (B, D, H', W')
                             │
     ┌───────────────────────┴───────────────────────┐
     │                                               │
     ▼ (Concat or Cross-Attention Fusion)            ▼
[Fused Latent Representation Z_fused]           [Task Decoder]
     ▲                                               │
     │ Z_radar (B, D, H', W')                        ▼
[Radar Encoder]                             [Segmentation Mask]
     │                                         (B, 2, 512, 512)
     │
┌────┴─────────────────┐
│   Sentinel-1 SAR     │
└──────────────────────┘
```

| ID | Model Architecture | Inputs | Purpose |
| :---: | :--- | :--- | :--- |
| **E1** | Simple UNet / ResNet-50 baseline | Optical only (`s2`) | Conventional vision baseline |
| **E2** | Frozen Prithvi-EO-2.0 100M-TL | Optical 6 bands (`s2`) | Test transferability of pretrained foundation features |
| **E3** | Fine-tuned Prithvi-EO-2.0 | Optical 6 bands (`s2`) | Domain adaptation on flood downstream task |
| **E4** | Radar Baseline | Radar only (`s1`) | Test SAR standalone performance in bad weather |
| **E5** | **Proposed Multimodal Fused Model** | **Optical (`s2`) + Radar (`s1`)** | **Core Research Contribution: Prove multimodality beats single sensors** |
| **E6** | Generalization Benchmark | Best model evaluated on `flood_bolivia_data.csv` | Cross-geographic transfer test |

### Fusion Implementations: Start Simple, Then Advance
1. **Feature Concatenation (Early/Mid-level):**
   $$Z_{\text{fused}} = \text{Conv}_{1\times 1}([Z_{\text{optical}} \,;\, Z_{\text{radar}}])$$
2. **Cross-Attention Fusion (Late-level):**
   Use SAR features as Query and Optical features as Key/Value to guide the model when optical data has high cloud obscuration.

---

## 📊 5. Standardized Evaluation Interface

Evaluate all validation and test epochs using Person 2's verified metric accumulator:

```python
from src.evaluation.metrics import MetricAccumulator

accumulator = MetricAccumulator(num_classes=2, ignore_index=-1)

model.eval()
with torch.no_grad():
    for batch in val_loader:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        mask = batch["mask"].to(device)

        # Forward pass (logits shape: B, 2, 512, 512)
        logits = model(s1, s2)
        preds = logits.argmax(dim=1)

        # Update metric accumulator
        accumulator.update(preds, mask)

# Compute epoch results dictionary
results = accumulator.compute()
print(f"Validation Mean IoU  : {results['mean_iou']:.4f}")
print(f"Validation Water IoU : {results['water_iou']:.4f}")
print(f"Validation Water F1  : {results['water_f1']:.4f}")
```

---

## 📦 6. What Person 1 Must Deliver Back to Person 2

When your models finish training, provide Person 2 with:
1. **Model Checkpoints:** Saved in `experiments/checkpoints/` (e.g. `best_prithvi_fused.pt`).
2. **Inference Function:** A standard wrapper function in `src/models/predict.py`:
   ```python
   def predict_chip(s1_tensor, s2_tensor) -> np.ndarray:
       """Accepts preprocessed tensors, returns (512, 512) binary mask."""
   ```
3. **Ablation Comparison Table:** Comparing E1, E2, E3, E4, and E5 on `water_iou` and `water_f1` for the final research report.

Person 2 will immediately wire your `predict_chip` function into the live **Streamlit Environmental Intelligence Dashboard** (`http://localhost:8501`) to generate real-time flood extent maps and automated environmental reports.

---

## 🚀 7. Person 1 Action Plan (Step-by-Step)

- [ ] **Step 1:** Download model weights for **Prithvi-EO-2.0-100M-TL** from Hugging Face (`ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL`).
- [ ] **Step 2:** Test the DataLoader with `python -c "from src.data.datasets import get_dataloader; l = get_dataloader('data/splits/flood_train_data.csv', 'data/raw/sen1floods11', batch_size=2); b = next(iter(l)); print(b['s1'].shape, b['s2'].shape, b['mask'].shape)"`.
- [ ] **Step 3:** Implement the frozen encoder baseline (E2) and train a lightweight segmentation head with `ignore_index=-1`.
- [ ] **Step 4:** Build the optical-SAR fusion module (E5) and log `water_iou` metrics.
- [ ] **Step 5:** Export the best checkpoint and hand off `predict_chip` to Person 2 for the interactive dashboard.
