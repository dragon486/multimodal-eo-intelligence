# 06. Repository Structure, Tooling & Environment Setup

## 1. Clean Codebase Architecture

The project must strictly separate raw geospatial data, model definitions, experiment configurations, and user interfaces.

```
foundation-vision-environmental-intelligence/
├── configs/
│   ├── config_data.yaml         # Dataset paths, band configurations, normalization constants
│   ├── config_prithvi.yaml      # Prithvi encoder hyperparameters, patch size, embedding dims
│   └── config_fusion.yaml       # Multimodal fusion parameters, learning rates, loss weights
├── data/
│   ├── raw/                     # Downloaded raw GeoTIFFs & archives (NEVER COMMIT TO GIT)
│   ├── processed/               # Standardized, co-registered 224x224 .npy / .pt chips
│   └── splits/                  # Spatial block train.json, val.json, test.json split manifests
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Inspect band distributions, cloud masks & SAR histograms
│   ├── 02_model_inference_sanity.ipynb  # Verify forward pass tensor shapes & feature maps
│   └── 03_evaluation_visualizer.ipynb   # Plot PR-curves, confusion matrices, and IoU comparisons
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py          # Automated STAC / URL downloaders with checksum validation
│   │   ├── preprocess.py        # CRS reprojection, SCL cloud masking, SAR calibration
│   │   └── datasets.py          # PyTorch Dataset & DataLoader implementations (TorchGeo base)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── foundation_encoder.py# Wrappers for Prithvi-EO-2.0, TerraMind & Clay
│   │   ├── fusion.py            # Gated concatenation & Bidirectional cross-attention modules
│   │   └── task_heads.py        # Multi-label MLP classifier & FPN / U-Net segmentation decoders
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # Main PyTorch training loop with AMP FP16 & gradient accumulation
│   │   ├── evaluate.py          # Metric evaluators (Macro-F1, mIoU, ECE, calibration)
│   │   └── losses.py            # Focal loss, Dice loss, multi-label binary cross-entropy
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py           # Clean inference API accepting raw rasters or preprocessed tensors
│   └── utils/
│       ├── __init__.py
│       ├── geo_utils.py         # GeoTIFF metadata preservation, affine transforms, georeferencing
│       └── metrics.py           # Custom TorchMetrics wrappers
├── app/
│   ├── dashboard.py             # Streamlit interactive application
│   └── components/
│       ├── map_viewer.py        # Leafmap / Folium interactive satellite map layers
│       └── metrics_panel.py     # Quantitative statistics & flood area calculation cards
├── tests/
│   ├── test_data_shapes.py      # Assert that data loaders conform to Data Contract
│   ├── test_forward_pass.py     # Assert that models output expected logits shape
│   └── test_metrics.py          # Unit test evaluation metrics against synthetic ground truth
├── reports/
│   ├── figures/                 # Exported charts, ROC/PR curves, spatial prediction maps
│   └── final_research_paper.pdf # Compiled report / manuscript
├── requirements.txt             # Locked Python dependencies
├── .gitignore                   # Excludes data/, checkpoints/, __pycache__/, .env
├── README.md                    # Project landing page & quickstart
└── LICENSE                      # Open-source license (Apache-2.0 / MIT)
```

---

## 2. Technology Stack & Dependencies

| Layer | Tools | Purpose |
| :--- | :--- | :--- |
| **Core Language** | Python 3.10+ | Primary language |
| **Deep Learning** | `torch`, `torchvision`, `torchaudio` | Model training & tensor operations |
| **Foundation Frameworks** | `terratorch`, `transformers`, `timm` | IBM/NASA TerraTorch & Hugging Face EO models |
| **Remote Sensing ML** | `torchgeo` | Geospatial datasets, samplers, and data transforms |
| **Geospatial Processing** | `rasterio`, `gdal`, `geopandas`, `shapely` | Reading GeoTIFFs, CRS reprojection, vector geometries |
| **Multidimensional Data** | `numpy`, `xarray`, `rioxarray` | Multi-band array manipulations |
| **Evaluation Metrics** | `torchmetrics`, `scikit-learn` | Precision, Recall, Macro-F1, mIoU, Confusion Matrix |
| **Visualization** | `matplotlib`, `seaborn`, `folium`, `leafmap` | Geospatial mapping & publication-quality figures |
| **Interactive App** | `streamlit` | Environmental intelligence dashboard |
| **Experiment Tracking** | `mlflow` or `wandb` | Hyperparameter logging & metric tracking |

---

## 3. Recommended Environment Setup (Windows / NVIDIA RTX 2060)

Execute in PowerShell or Windows Terminal:

```powershell
# 1. Create dedicated Conda environment with Python 3.10
conda create -n geo-foundation python=3.10 -y
conda activate geo-foundation

# 2. Install PyTorch with CUDA 11.8 or 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. Install GDAL and Rasterio safely via conda-forge (prevents DLL conflict on Windows)
conda install -c conda-forge gdal rasterio geopandas -y

# 4. Install remaining remote sensing & foundation model libraries
pip install torchgeo terratorch transformers timm xarray rioxarray torchmetrics scikit-learn streamlit leafmap folium matplotlib seaborn pyproj
```

---

## 4. Git Version Control & Checkpoint Policy

To prevent committing multi-gigabyte satellite rasters or model weights:

Create or maintain `.gitignore`:
```gitignore
# Data and models (NEVER commit large assets)
data/raw/
data/processed/
*.tif
*.tiff
*.h5
*.hdf
*.nc
*.tar.gz
*.zip
*.pt
*.pth
*.ckpt

# Python & Environment
__pycache__/
*.pyc
.env
.venv/
env/
dist/
build/
.ipynb_checkpoints/
.pytest_cache/
mlruns/
wandb/
```

---

## 5. Final Project Deliverables Checklist (Acceptance Gating)

Use this checklist to track completion across the 8 required milestones:

- [ ] **1. Research Specification:** Clearly stated research question, cloud-robust hypothesis, and related work review.
- [ ] **2. Dataset Documentation:** Complete inventory of BigEarthNet v2.0, Sen1Floods11, and HLS, including licences and spatial split manifests.
- [ ] **3. Reproducible Pipeline:** Documented data loaders conforming to the Shared Data Contract and reproducible config files.
- [ ] **4. Experimental Results:** Comprehensive results for experiments E1 through E6, complete with Macro-F1, mIoU, and class-wise breakdowns.
- [ ] **5. Controlled Ablations:** Validated cloud-obscuration stress test (0% to 100% cloud degradation) proving the value of SAR fusion.
- [ ] **6. Georeferenced Artifacts:** Exported GeoTIFF classification/flood maps with intact coordinate reference metadata.
- [ ] **7. Environmental Intelligence Dashboard:** Working Streamlit app supporting interactive map viewing, area calculations (hectares), and confidence overlays.
- [ ] **8. Final Research Manuscript / Report:** Comprehensive report detailing methodology, experimental findings, failure modes, and ethical considerations.
