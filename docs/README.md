# Foundation Vision Models for Multimodal Remote Sensing and Environmental Intelligence
## Master Documentation & Research Knowledge Base

Welcome to the central research and engineering documentation repository for **Foundation Vision Models for Multimodal Remote Sensing and Environmental Intelligence**. 

This documentation serves as the single source of truth for both team members (**Person 1: Foundation Models & Multimodal Learning** and **Person 2: Geospatial Data & Environmental Intelligence**), providing deep architectural designs, dataset specifications, preprocessing protocols, fusion mechanics, evaluation frameworks, and collaborative contracts.

---

## 📑 Documentation Structure

```
docs/
├── README.md                                 # Master overview, quick navigation & executive summary
├── 01_project_overview_and_architecture.md   # Research scope, objectives, end-to-end architecture & hypotheses
├── 02_datasets_and_preprocessing.md          # Data specifications (S1, S2, HLS), preprocessing & leakage-proof splits
├── 03_foundation_models_and_fusion.md        # Prithvi-EO-2.0, TerraMind, Clay, fusion math & 6GB VRAM strategies
├── 04_training_and_evaluation.md             # Experiments matrix (E1-E6), ablation design, metrics & cloud-robust tests
├── 05_team_division_and_contracts.md         # Person 1 & Person 2 task matrix, data contracts & 8-week timeline
└── 06_repository_and_tooling.md              # Codebase layout, dependencies, environment setup & reproducibility
```

---

## 🧭 Document Guide

| Document | Key Topics Covered | Primary Owner |
| :--- | :--- | :--- |
| **[01. Overview & Architecture](01_project_overview_and_architecture.md)** | Core research goals, cloud-robust hypothesis, 6-stage end-to-end system design, environmental intelligence deliverables. | Shared |
| **[02. Datasets & Preprocessing](02_datasets_and_preprocessing.md)** | BigEarthNet v2.0, NASA HLS, Sen1Floods11, BigEarthNet.txt, coordinate alignment, CRS reprojection, cloud masking, spatial blocking. | **Person 2** |
| **[03. Foundation Models & Fusion](03_foundation_models_and_fusion.md)** | Prithvi-EO-2.0 100M-TL, TerraMind, Clay, feature concatenation, cross-attention fusion, 6GB VRAM (RTX 2060) optimization tactics. | **Person 1** |
| **[04. Training & Evaluation](04_training_and_evaluation.md)** | Baseline setup, E1–E6 experiment matrix, ablation studies, missing-modality testing, Macro-F1, mIoU, calibration. | Shared |
| **[05. Team Division & Contracts](05_team_division_and_contracts.md)** | Concrete roles, the Strict Tensor Data Contract (shapes, band orders, normalization), 8-week execution roadmap. | Shared |
| **[06. Repository & Tooling](06_repository_and_tooling.md)** | Modular folder structure, PyTorch/TerraTorch/TorchGeo/Rasterio dependencies, configuration templates, checklist. | Shared |

---

## 🎯 Quick Research Summary

* **Central Research Hypothesis**: Does fusing pretrained optical (Sentinel-2/HLS) and synthetic aperture radar (Sentinel-1 SAR) foundation-model representations provide statistically significant robustness improvements for environmental intelligence (land-cover classification, flood segmentation, and change detection) under cloud obscuration and sensor degradation compared to single-modality baselines?
* **Primary Foundation Model**: `Prithvi-EO-2.0-100M-TL` (IBM/NASA geospatial ViT pretrained on HLS via Masked Autoencoding).
* **Multimodal Complement**: Sentinel-1 SAR dual-polarization ($\text{VV} + \text{VH}$) paired with optical Sentinel-2/HLS.
* **Core Benchmark Datasets**:
  1. **BigEarthNet v2.0** (549,488 paired S1/S2 image patches for land-cover classification).
  2. **Sen1Floods11** (4,831 paired georeferenced chips across 11 global flood events for flood-water segmentation).
  3. **NASA HLS** (Harmonized Landsat-Sentinel 30m time-series for temporal environmental dynamics).
* **Compute Target**: NVIDIA RTX 2060 (6 GB VRAM) local prototyping with cloud GPU burst for scale-up training.
