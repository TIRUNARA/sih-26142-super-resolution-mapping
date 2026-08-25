# 🛰️ SIH 26142: Deep Learning Based Super Resolution Mapping (SRM)

**Sponsoring Organization:** National Technical Research Organisation (NTRO)  
**Theme:** Space Technology / Smart India Hackathon 2026  
**Problem Statement ID:** SIH26142  

---

## 🎯 Overview
Medium-resolution satellite imagery (10m - 30m, e.g., Sentinel-2, Landsat-8/9) offers excellent temporal revisit times and global coverage, but lacks spatial clarity for micro-land cover analysis, urban monitoring, and disaster damage assessment. 

This repository implements a **Geospatial Deep Learning Pipeline for Super Resolution Mapping (SRM)**, converting medium-resolution multi-spectral bands into high-resolution land cover maps and enhanced spatial imagery while solving the **mixed pixel problem**.

---

## 🏗️ Technical Architecture & Pipeline

1. **Geospatial Ingestion Engine**: Handlers for GeoTIFF multi-spectral bands (RGB + NIR + SWIR), raster alignment, and normalisation.
2. **Deep Learning Core**: 
   - **SwinIR / RCAN Backbone**: Residual Channel Attention & Swin Transformer architecture optimized for satellite spectral preservation.
   - **Sub-Pixel Mapping (SPM)**: Spatial allocation of land cover classes within mixed pixels.
3. **Inference & Tiling API**: FastAPI async server streaming tiled predictions over large GeoTIFF rasters.
4. **Interactive Geospatial Dashboard**: Web application for split-screen before/after comparison, spectral profile inspection, and GeoTIFF downloads.

---

## ⚡ Project Structure
```
sih-26142-super-resolution-mapping/
├── architecture.json         # Visual system blueprint
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── src/
    ├── api/                  # FastAPI inference endpoints
    ├── data/                 # GeoTIFF loader & patch generator
    ├── models/               # PyTorch Super-Resolution neural networks
    └── ui/                   # Web comparison workbench
```

---
*Built with precision for SIH 2026 by TIRUNARA & Integrity.*
