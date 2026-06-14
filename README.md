# Road Damage Detection and Segmentation using YOLOv8, SAHI, and U-Net

## Overview

This B.Tech thesis project focuses on the automated detection and segmentation of road damage using state-of-the-art deep learning techniques. The proposed framework combines YOLOv8 for object detection, SAHI (Slicing Aided Hyper Inference) for improved small-object detection, and U-Net for pixel-level segmentation of road defects.

The objective is to develop an efficient and scalable system capable of detecting and segmenting road damage such as cracks and potholes from road-view images captured using vehicle-mounted cameras.

---

## Problem Statement

Road damage such as cracks and potholes negatively impacts transportation safety, vehicle performance, and infrastructure maintenance costs. Traditional manual inspection methods are labor-intensive, time-consuming, subjective, and difficult to scale across large road networks.

Deep learning-based computer vision approaches offer an opportunity to automate road condition monitoring. However, detecting small and irregular road defects remains challenging due to:

* Small object sizes relative to image dimensions
* Variations in lighting and weather conditions
* Road texture diversity
* Shadows and occlusions
* Large intra-class variation of damage types

This project aims to address these challenges by combining YOLOv8, SAHI, and U-Net into a unified road damage analysis pipeline.

---

## Proposed Pipeline

```text
Road Image
    ↓
SAHI Slicing
    ↓
YOLOv8 Detection
    ↓
Bounding Boxes
    ↓
U-Net Segmentation
    ↓
Pixel-Level Damage Mask
```

---

## Dataset

### Detection Dataset

**RDD2022 (Road Damage Detection 2022)**

The RDD2022 dataset contains road damage images collected from multiple countries and annotated with bounding boxes for various damage categories.

Classes:

| Class ID | Damage Type        |
| -------- | ------------------ |
| 0        | Longitudinal Crack |
| 1        | Transverse Crack   |
| 2        | Alligator Crack    |
| 3        | Other Corruption   |
| 4        | Pothole            |

### Segmentation Dataset (Planned)

* Crack500
* DeepCrack

---

## Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* SAHI
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Segmentation Models PyTorch
* Google Colab
* Kaggle Notebooks
* VS Code
* GitHub

---

## Repository Structure

```text
road-damage-thesis/
│
├── datasets/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_yolov8_baseline.ipynb
│   └── 03_yolov8s_baseline.ipynb
│
├── detection/
│   ├── checkpoints/
│   ├── configs/
│   ├── training/
│   ├── inference/
│   └── yolov8/
│
├── sahi/
│   ├── sliced_inference/
│   ├── experiments/
│   └── comparison_results/
│
├── segmentation/
│   ├── unet/
│   ├── datasets/
│   ├── training/
│   ├── masks/
│   └── checkpoints/
│
├── experiments/
│   └── baseline_vs_sahi/
│
├── visualizations/
│   ├── predictions/
│   ├── comparison_figures/
│   ├── segmentation_masks/
│   └── thesis_images/
│
├── reports/
│   ├── literature_review/
│   ├── methodology/
│   ├── results/
│   ├── comparison/
│   └── final_report/
│
├── presentations/
│
├── requirements.txt
├── README.md
└── thesis_roadmap.md
```

---

## Exploratory Data Analysis (Completed)

The RDD2022 dataset was analyzed to better understand the distribution and characteristics of road damage instances.

Completed analyses include:

* Class Distribution Analysis
* Bounding Box Distribution Analysis
* Object Size Distribution Analysis
* Country-wise Dataset Exploration
* Sample Annotation Visualization
* Damage Category Exploration

Key observations:

* Longitudinal cracks are the most frequent damage category.
* Potholes are the least represented class.
* A significant proportion of damage instances occupy a very small fraction of the image area.
* Small object prevalence motivates the use of SAHI-based sliced inference.

---

## Baseline Detection Experiments

Two baseline object detection models were trained using identical hyperparameters:

### YOLOv8n Baseline

* Epochs: 20
* Image Size: 640 × 640
* Batch Size: 16

Results:

| Metric       | Value |
| ------------ | ----: |
| Precision    | 0.579 |
| Recall       | 0.516 |
| mAP@0.5      | 0.539 |
| mAP@0.5:0.95 | 0.285 |

---

### YOLOv8s Baseline

* Epochs: 20
* Image Size: 640 × 640
* Batch Size: 16

Results:

| Metric       | Value |
| ------------ | ----: |
| Precision    | 0.626 |
| Recall       | 0.553 |
| mAP@0.5      | 0.592 |
| mAP@0.5:0.95 | 0.317 |

---

## Baseline Model Comparison

| Model   | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ------- | --------: | -----: | ------: | -----------: |
| YOLOv8n |     0.579 |  0.516 |   0.539 |        0.285 |
| YOLOv8s |     0.626 |  0.553 |   0.592 |        0.317 |

### Key Findings

* YOLOv8s outperformed YOLOv8n across all evaluation metrics.
* The most significant improvement was observed in mAP metrics.
* Pothole detection remained the most challenging category.
* Other Corruption achieved the highest detection accuracy.
* Most errors resulted from missed detections rather than incorrect classifications.
* Small object detection remains a key challenge, motivating the integration of SAHI.

---

## Current Progress

### Completed

* Literature Review
* Research Paper Analysis
* Project Structure Setup
* Dataset Acquisition (RDD2022)
* Exploratory Data Analysis
* YOLOv8n Baseline Training and Evaluation
* YOLOv8s Baseline Training and Evaluation
* Baseline Model Comparison
* Repository Organization and Experiment Tracking

### In Progress

* SAHI Integration for Small Object Detection
* Baseline vs SAHI Comparative Evaluation

### Upcoming

* U-Net Segmentation Pipeline
* Crack500 Dataset Integration
* DeepCrack Dataset Evaluation
* Detection and Segmentation Pipeline Integration
* Experimental Analysis
* Thesis Documentation
* Final Presentation Preparation

---

## Evaluation Metrics

### Detection Metrics

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* Confusion Matrix Analysis
* Precision-Recall Curves

### Segmentation Metrics

* Dice Score
* Intersection over Union (IoU)
* Pixel Accuracy

---

## Future Work

Planned experiments include:

1. SAHI + YOLOv8n
2. SAHI + YOLOv8s
3. Baseline vs SAHI Performance Comparison
4. Crack Segmentation using U-Net
5. Combined Detection and Segmentation Pipeline
6. Real-world Deployment Considerations

---

## Thesis Goal

To develop an efficient deep learning framework capable of accurately detecting and segmenting road damage, with a particular focus on improving the detection of small cracks and potholes through SAHI-based sliced inference and pixel-level segmentation using U-Net.

The final system aims to support automated road condition monitoring and intelligent infrastructure maintenance.
