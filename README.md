# Multimodal Earth Observation & Environmental Intelligence

An end-to-end foundation vision system combining Sentinel-1 SAR (Radar) and Sentinel-2 Optical (Multispectral) imagery for flood segmentation, land-cover mapping, and environmental intelligence.

---

## 👥 Team Roles & Responsibilities

- **Person 1 (Model Lead):** Pretrained foundation encoders (Prithvi-EO-2.0, TerraMind), multimodal fusion architectures (concatenation / cross-attention), fine-tuning, and model training.
- **Person 2 (Data & Evaluation Lead):** Satellite data acquisition (Sen1Floods11, BigEarthNet v2.0), geospatial preprocessing, spatial leakage prevention, evaluation benchmark metrics (IoU, mIoU, Macro-F1), visualization, and Streamlit environmental intelligence dashboard.

---

## 🚀 Quickstart for Person 2

### 1. Download Official Split CSVs & Sample Imagery
```bash
# Download official GCS split definitions
python src/data/download_sen1floods11.py

# Download a lightweight initial subset for testing (e.g. 5 chips per split)
python src/data/download_sen1floods11.py --subset --num 5
```

### 2. Audit Dataset Splits for Spatial Leakage
```bash
python src/data/split.py
```

### 3. Inspect GeoTIFF Rasters & Label Distribution
```bash
# Inspect SAR raster
python src/data/inspect_raster.py data/raw/test_s1.tif

# Inspect Optical multispectral raster
python src/data/inspect_raster.py data/raw/test_s2.tif

# Inspect ground truth mask with class frequency breakdown
python src/data/inspect_raster.py data/raw/test_label.tif --mask
```

### 4. Run End-to-End Pipeline Verification
```bash
python test_pipeline.py
```
Outputs qualitative 4-panel evaluation plots in `reports/visualizations/`.

### 5. Launch Environmental Intelligence Dashboard
```bash
streamlit run src/app/dashboard.py
```

---

## 📁 Repository Structure

```
fv_model/
├── configs/                # Experiment & dataset configurations
├── data/
│   ├── raw/                # Downloaded raw GeoTIFFs (gitignored)
│   ├── processed/          # Preprocessed tensors / chips
│   └── splits/             # Official train / val / test CSVs
├── notebooks/              # Jupyter exploration notebooks
├── src/
│   ├── data/
│   │   ├── download_sen1floods11.py  # GCS dataset downloader
│   │   ├── inspect_raster.py         # GeoTIFF metadata & class stats
│   │   ├── preprocess.py             # Radiometric calibration & normalization
│   │   ├── datasets.py               # PyTorch Dataset & DataLoader
│   │   └── split.py                  # Spatial leakage & regional audit
│   ├── evaluation/
│   │   ├── metrics.py                # IoU, mIoU, F1, confusion matrix
│   │   └── visualize.py              # 4-panel multi-sensor comparison plots
│   └── app/
│       └── dashboard.py              # Streamlit environmental interface
├── reports/
│   ├── DATA_CONTRACT.md    # Shared interface specification for Person 1
│   ├── dataset_report.md   # Dataset properties and statistical breakdown
│   └── visualizations/     # Rendered output figures
├── requirements.txt
├── test_pipeline.py        # End-to-end integration test
└── README.md
```
