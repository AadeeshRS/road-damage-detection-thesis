# Road Damage Detection and Crack Segmentation using YOLOv8, YOLO11, Faster R-CNN, SAHI, U-Net, and DeepLabV3+

## Overview

This repository contains the implementation, experiments, and results for a B.Tech thesis focused on automated road damage detection and crack segmentation using deep learning.

The project investigates the effectiveness of one-stage and two-stage object detection models, including YOLOv8, YOLO11, and Faster R-CNN, along with SAHI (Slicing Aided Hyper Inference) for improving small-object detection performance. For pixel-level crack extraction, semantic segmentation models including U-Net and DeepLabV3+ were evaluated.

The primary objective is to develop an efficient framework capable of identifying and localizing road defects such as longitudinal cracks, transverse cracks, alligator cracks, potholes, and other surface deteriorations from road-view imagery.

---
## Sample Road Damage Detection

<p align="center">
  <img src="reports/results/detection/sahi/yolov11n/thesis_examples/China_MotorBike_000526.png" width="500">
</p>

<p align="center">
  <em>Example road damage detections produced by YOLO11n on the RDD2022 dataset.</em>
</p>

---
## Project Highlights

### Detection Models

- YOLOv8n
- YOLOv8s
- YOLOv8m
- YOLO11n
- YOLO11s
- YOLO11m
- Faster R-CNN (ResNet50-FPN)

### SAHI Enhanced Inference

- SAHI + YOLOv8n
- SAHI + YOLOv8s
- SAHI + YOLOv8m
- SAHI + YOLO11n
- SAHI + YOLO11s
- SAHI + YOLO11m
- Exploratory: Faster R-CNN + SAHI

### Segmentation Models

- U-Net (ResNet34 Encoder)
- DeepLabV3+ (ResNet34 Encoder)

### Datasets

- RDD2022 (Road Damage Detection)
- Crack500 (Road Crack Segmentation)

---

## Detection Results

### Baseline Model Comparison

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|--------|-----------|--------|---------|------------|
| YOLOv8n | 0.579 | 0.516 | 0.539 | 0.285 |
| YOLOv8s | 0.626 | 0.553 | 0.592 | 0.317 |
| **YOLOv8m** | **0.677** | **0.590** | **0.633** | **0.344** |
| YOLO11n | 0.593 | 0.514 | 0.544 | 0.287 |
| YOLO11s | 0.613 | 0.540 | 0.575 | 0.309 |
| YOLO11m | 0.631 | 0.546 | 0.587 | 0.314 |
| Faster R-CNN | N/A | 0.427 | 0.536 | 0.253 |

### Key Observations

- YOLOv8m achieved the highest overall detection performance.
- YOLO11 models provided competitive results but did not outperform YOLOv8m on the RDD2022 dataset.
- Faster R-CNN served as a strong two-stage baseline but underperformed compared to larger YOLO variants.

---

## SAHI Evaluation

To improve small-object detection, SAHI (Slicing Aided Hyper Inference) was applied to all detection models.

### YOLO + SAHI Results

| Model | Baseline Detections | SAHI Detections | Improvement |
|--------|-------------------:|----------------:|-------------:|
| YOLOv8n | 3773 | 4564 | 20.96% |
| **YOLOv8s** | 4237 | 5475 | **29.22%** |
| YOLOv8m | 4551 | 5321 | 16.92% |
| YOLO11n | 3612 | 4496 | 24.47% |
| YOLO11s | 4016 | 4826 | 20.17% |
| YOLO11m | 3986 | 4722 | 18.46% |

### Key Observations

- SAHI consistently improved detection counts across all YOLO models.
- YOLOv8s achieved the largest improvement (+29.22%).
- YOLO11n demonstrated the highest improvement among YOLO11 variants (+24.47%).
- Sliced inference significantly improved the localization of small and distant road defects.

---
## Qualitative SAHI Example

<p align="center">
  <img src="reports/results/detection/sahi/yolov8m/top_gain_comparisons/top_gain_3.png" width="900">
</p>

<p align="center">
  <em>Comparison between standard YOLOv8m inference and SAHI-enhanced inference. Sliced inference enables the detection of additional small and distant road defects while improving localization performance.</em>
</p>

---

## Exploratory Study: Faster R-CNN + SAHI

An exploratory analysis was conducted to investigate the effectiveness of SAHI on a two-stage detector.

### Experimental Results

| Model | Images Evaluated | Baseline Detections | SAHI Detections | Improvement |
|--------|-----------------|-------------------:|----------------:|-------------:|
| Faster R-CNN | 2000 | 6899 | 11256 | 63.15% |

### Discussion

Although Faster R-CNN + SAHI produced a substantial increase in detected road damage instances, qualitative analysis revealed a significant rise in:

- False positives
- Overlapping predictions
- Duplicate detections
- Class confusion in complex road textures

This behavior differed considerably from the more stable improvements observed with YOLO-based models.

Consequently, Faster R-CNN + SAHI is treated as an exploratory study rather than a primary contribution of this work.

---

## Segmentation Results

Crack segmentation experiments were conducted on the Crack500 dataset.

### Model Comparison

| Model | Encoder | Epochs | Batch Size | Dice Score | IoU Score |
|--------|----------|--------|------------|------------|-----------|
| **U-Net** | ResNet34 | 30 | 8 | **0.7533** | **0.6118** |
| DeepLabV3+ | ResNet34 | 30 | 8 | 0.7389 | 0.5931 |

### Key Observations

- U-Net achieved the highest segmentation performance.
- DeepLabV3+ produced competitive results but slightly underperformed U-Net.
- U-Net was selected as the final segmentation architecture for this project.

---
## Qualitative Segmentation Comparison

<p align="center">
  <img src="reports/results/final_figures/figure_14_segmentation_comparison.png" width="1000">
</p>

<p align="center">
  <em>Qualitative comparison between U-Net and DeepLabV3+ on the Crack500 dataset. U-Net achieved superior Dice and IoU scores and produced more accurate crack boundaries.</em>
</p>

---

## Dataset Information

### RDD2022

The Road Damage Detection 2022 (RDD2022) dataset contains road images collected from multiple countries, including:

- India
- Japan
- Norway
- United States
- Czech Republic

### Detection Classes

| ID | Class |
|----|--------|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Other Corruption |
| 4 | Pothole |

### Crack500

Crack500 is a road crack segmentation dataset designed for pixel-level crack extraction tasks.

---

## Methodology

### Detection Pipeline

```text
Road Image
      │
      ▼
YOLO / Faster R-CNN
      │
      ▼
Bounding Box Detection
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
Detection Model Inference
      │
      ▼
Prediction Merging
      │
      ▼
Enhanced Small-Object Detection
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
Thesis/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── datasets/
│
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
│   ├── 10_segmentation_comparison.ipynb
│   ├── 11_yolov11n_baseline.ipynb
│   ├── 12_yolov11s_baseline.ipynb
│   ├── 13_yolov11m_baseline.ipynb
│   ├── 14_faster_rcnn.ipynb
│   ├── 15_yolov11n_sahi.ipynb
│   ├── 16_yolov11s_sahi.ipynb
│   ├── 17_yolov11m_sahi.ipynb
│   └── 18_faster_rcnn_sahi.ipynb
│
├── presentations/
│
└── reports/
    ├── final_report/
    ├── methodology/
    ├── literature_review/
    └── results/
        ├── detection/
        ├── segmentation/
        ├── eda/
        └── final_figures/
```

---

## Key Findings

### Detection

- YOLOv8m achieved the highest overall detection performance.
- SAHI consistently improved small-object detection across all YOLO models.
- YOLOv8s + SAHI achieved the largest improvement (+29.22%).
- Faster R-CNN provided a competitive two-stage baseline.
- Faster R-CNN + SAHI produced higher sensitivity but substantially increased false positives.

### Segmentation

- U-Net outperformed DeepLabV3+ on Crack500.
- Crack segmentation performance exceeded 0.75 Dice Score.
- ResNet34 proved to be an effective encoder backbone for both architectures.

### Failure Cases

Common sources of false positives include:

- Vegetation and roadside plants
- Shadows
- Snow-covered road regions
- High-contrast road textures
- Lane markings and worn paint
- Complex asphalt patterns

These cases highlight the challenges of robust road damage detection under varying environmental conditions.

---

## Experimental Artifacts

The repository includes:

- Training curves
- Confusion matrices
- Precision-Recall curves
- F1 score curves
- SAHI comparison figures
- Segmentation predictions
- Failure case analysis
- Thesis-ready visualizations
- Final quantitative comparison tables

All selected figures are stored under:

```text
reports/results/final_figures/
```

---

## Project Status

- ✅ Exploratory Data Analysis
- ✅ YOLOv8 Training and Evaluation
- ✅ YOLO11 Training and Evaluation
- ✅ Faster R-CNN Training and Evaluation
- ✅ SAHI Evaluation
- ✅ Exploratory Faster R-CNN + SAHI Analysis
- ✅ U-Net Segmentation
- ✅ DeepLabV3+ Segmentation
- ✅ Experimental Comparison
- 🚧 Thesis Writing and Documentation

---

## Future Work

Potential extensions include:

- Transformer-based object detectors
- Vision Transformer segmentation models
- Real-time deployment on edge devices
- Additional road crack segmentation datasets
- Domain adaptation across countries
- Multi-modal road inspection systems
- Integration with intelligent road maintenance frameworks

---

## Author

**Aadeesh Ranjan**

B.Tech Computer Science and Engineering

Road Damage Detection and Crack Segmentation Thesis Project