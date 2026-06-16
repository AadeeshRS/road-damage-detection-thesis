# Road Damage Detection and Crack Segmentation using YOLOv8, SAHI, U-Net, and DeepLabV3+

## Overview

This repository contains the implementation and experimental results for a B.Tech thesis focused on automated road damage detection and crack segmentation using deep learning.

The project investigates the effectiveness of YOLOv8 object detectors for road damage detection, SAHI (Slicing Aided Hyper Inference) for improving small-object detection, and semantic segmentation models including U-Net and DeepLabV3+ for pixel-level crack extraction.

The primary objective is to develop an efficient framework capable of identifying and localizing road defects such as longitudinal cracks, transverse cracks, alligator cracks, potholes, and other surface deteriorations from road-view imagery.

---

## Project Highlights

### Detection Models

* YOLOv8n
* YOLOv8s
* YOLOv8m

### SAHI Enhanced Inference

* SAHI + YOLOv8n
* SAHI + YOLOv8s
* SAHI + YOLOv8m

### Segmentation Models

* U-Net
* DeepLabV3+

### Datasets

* RDD2022 (Detection)
* Crack500 (Segmentation)

---

## Detection Results

### Baseline Model Comparison

| Model   | Precision |    Recall |    mAP@50 | mAP@50-95 |
| ------- | --------: | --------: | --------: | --------: |
| YOLOv8n |     0.579 |     0.516 |     0.539 |     0.285 |
| YOLOv8s |     0.626 |     0.553 |     0.592 |     0.317 |
| YOLOv8m | **0.677** | **0.590** | **0.633** | **0.344** |

### Key Observation

Increasing model capacity consistently improved road damage detection performance. YOLOv8m achieved the best overall detection performance across all evaluation metrics.

---

## SAHI Evaluation

To improve detection of small and distant road defects, SAHI was applied to all trained YOLOv8 models.

### Detection Count Comparison

| Model   | Baseline Detections | SAHI Detections | Improvement |
| ------- | ------------------: | --------------: | ----------: |
| YOLOv8n |                3773 |            4564 |      20.96% |
| YOLOv8s |                4237 |            5475 |  **29.22%** |
| YOLOv8m |                4551 |            5321 |      16.92% |

### Key Observation

SAHI consistently improved the number of detected road damage instances by slicing large images into smaller overlapping patches before inference.

The highest gain was observed with YOLOv8s, which achieved a 29.22% increase in detected road defects.

---

## Segmentation Results

Crack segmentation experiments were conducted on the Crack500 dataset using U-Net and DeepLabV3+.

### Model Comparison

| Model      | Dice Score |  IoU Score |
| ---------- | ---------: | ---------: |
| U-Net      | **0.7533** | **0.6118** |
| DeepLabV3+ |     0.7389 |     0.5931 |

### Key Observation

U-Net achieved the highest segmentation performance and was selected as the best-performing segmentation model for this project.

---

## Dataset Information

### RDD2022

Road Damage Detection 2022 dataset containing road images collected from multiple countries.

### Classes

| ID | Class              |
| -- | ------------------ |
| 0  | Longitudinal Crack |
| 1  | Transverse Crack   |
| 2  | Alligator Crack    |
| 3  | Other Corruption   |
| 4  | Pothole            |

### Crack500

Road crack segmentation dataset used for pixel-level damage extraction experiments.

---

## Methodology

### Detection Pipeline

```text
Road Image
      │
      ▼
YOLOv8 Detection
      │
      ▼
Bounding Boxes
      │
      ▼
Road Damage Classification
```

### SAHI Detection Pipeline

```text
Road Image
      │
      ▼
Image Slicing
      │
      ▼
YOLOv8 Inference
      │
      ▼
Prediction Merging
      │
      ▼
Enhanced Detection Results
```

### Segmentation Pipeline

```text
Road Image
      │
      ▼
U-Net / DeepLabV3+
      │
      ▼
Pixel-Level Crack Mask
```

---

## Repository Structure

```text
.
├── datasets/
├── detection/
│   └── checkpoints/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_yolov8n_baseline.ipynb
│   ├── 03_yolov8s_baseline.ipynb
│   ├── 04_sahi_yolov8n.ipynb
│   ├── 05_sahi_yolov8s.ipynb
│   ├── 06_yolov8m_baseline.ipynb
│   ├── 07_sahi_yolov8m.ipynb
│   ├── 08_unet_crack500.ipynb
│   ├── 09_deeplabv3plus_crack500.ipynb
│   └── 10_segmentation_comparison.ipynb
│
├── reports/
│   └── results/
│       ├── detection/
│       ├── segmentation/
│       ├── eda/
│       └── final_figures/
│
└── README.md
```

---

## Key Findings

### Detection

* YOLOv8m achieved the best baseline detection performance.
* SAHI significantly improved small-object detection across all YOLO models.
* YOLOv8s + SAHI achieved the largest improvement (+29.22%).

### Segmentation

* U-Net outperformed DeepLabV3+ on Crack500.
* Crack segmentation performance exceeded 0.75 Dice Score.

### Failure Cases

Common sources of false positives include:

* Vegetation and roadside plants
* Shadows
* Snow-covered regions
* High-contrast road textures

These cases highlight the challenges of road damage detection under varying environmental conditions.

---

## Experimental Artifacts

The repository includes:

* Training curves
* Confusion matrices
* Precision-Recall curves
* SAHI comparison figures
* Segmentation predictions
* Failure case analysis
* Final thesis-ready figures

All visual results are stored under:

```text
reports/results/final_figures/
```

---

## Project Status

* ✅ Exploratory Data Analysis
* ✅ YOLOv8n Training and Evaluation
* ✅ YOLOv8s Training and Evaluation
* ✅ YOLOv8m Training and Evaluation
* ✅ SAHI Evaluation
* ✅ U-Net Segmentation
* ✅ DeepLabV3+ Segmentation
* ✅ Experimental Comparison
* 🚧 Thesis Writing and Documentation

---

## Future Work

Potential extensions include:

* Transformer-based object detectors
* Real-time deployment on edge devices
* Additional crack segmentation datasets
* Domain adaptation across countries
* Integration with intelligent road maintenance systems

---

## Author

**Aadeesh Ranjan**

B.Tech Computer Science and Engineering

Road Damage Detection and Segmentation Thesis Project
