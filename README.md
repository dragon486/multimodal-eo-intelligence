# Multimodal Earth Observation & Environmental Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2Bcu121-ee4c2c.svg)](https://pytorch.org/)
[![GPU Accelerated](https://img.shields.io/badge/CUDA-RTX_2060_Verified-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Tests Passing](https://img.shields.io/badge/Tests-8%2F8_Passed-success.svg)](tests/)
[![Docs](https://img.shields.io/badge/Docs-Complete_Knowledge_Base-blueviolet.svg)](docs/README.md)

An end-to-end research framework combining **Sentinel-2 Multispectral Optical** and **Sentinel-1 C-Band SAR (Radar)** using **Geospatial Foundation Vision Models** for cloud-robust flood disaster mapping and environmental intelligence.

---

## 👥 Integrated Two-Person Engineering Topology

This repository seamlessly combines:
* **Person 1 (Foundation Models & Multimodal Learning Lead):**
  * `src/models/foundation_encoder.py`: Multi-spectral optical ViT (Prithvi-compatible) & radar SAR encoders with dynamic 2D bicubic positional interpolation.
  * `src/models/fusion.py`: Bidirectional Cross-Attention, Gated Concatenation, and Cloud-Adaptive Gating.
  * `src/models/task_heads.py`: Multi-label land-cover head & multi-resolution segmentation decoders.
  * `src/models/predict.py`: Standardized `predict_chip(s1, s2)` inference function.
  * `src/training/train_flood.py`: Mixed-precision (AMP FP16) training engine with class imbalance weighting and gradient accumulation.
* **Person 2 (Geospatial Data & Environmental Intelligence Lead):**
  * `data/splits/`: Official Sen1Floods11 splits (`train`, `val`, `test`) + 100% isolated held-out `flood_bolivia_data.csv`.
  * `src/data/download_sen1floods11.py`: Automated GCS chip downloader.
  * `src/data/preprocess.py`: SAR decibel clipping (`[-35 dB, +10 dB]`), optical reflectance scaling (`10,000`), and mask sanitization.
  * `src/data/class_balance.py`: Empirical class balance auditor (`[0.5566, 4.9183]` loss weights).
  * `src/data/split.py`: Spatial leakage and regional integrity verification tool.
  * `src/evaluation/metrics.py`: `MetricAccumulator` and `compute_segmentation_metrics` (with `ignore_index=-1`).
  * `src/evaluation/visualize.py`: Multi-panel SAR composite, RGB optical, and prediction plotting.
  * `src/app/dashboard.py`: Interactive Streamlit environmental intelligence dashboard.

---

## 🚀 Quickstart

### 1. Activate Environment
```powershell
.\.venv\Scripts\activate
```

### 2. Run Test Suite (8/8 Verified)
```powershell
pytest tests -v
```

### 3. Audit Dataset Splits for Zero Spatial Leakage
```powershell
python -m src.data.split data/splits
```

### 4. Train Multimodal Flood Model (Experiment E5)
```powershell
python src/training/train_flood.py --exp E5 --mode multimodal --epochs 5 --batch_size 2
```

### 5. Launch Interactive Environmental Dashboard
```powershell
streamlit run src/app/dashboard.py
```
Visit `http://localhost:8501` to view multi-sensor overlays, test the neural model live, and inspect quantitative flooded area reports.
