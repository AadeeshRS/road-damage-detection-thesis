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
│   ├── 02_yolov8n_training.ipynb
│   ├── 03_yolov8s_training.ipynb
│   ├── 04_sahi_yolov8n.ipynb
│   └── 05_sahi_yolov8s.ipynb
│
├── visualizations/
│   ├── comparison_figures/
│   ├── predictions/
│   ├── segmentation_masks/
│   └── thesis_images/
│
├── reports/
│   ├── literature_review/
│   ├── methodology/
│   ├── results/
│   │   ├── yolov8n/
│   │   ├── yolov8s/
│   │   └── sahi/
│   │       ├── yolov8n/
│   │       └── yolov8s/
│   └── final_report/
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

### Key Observations

* Longitudinal cracks are the most frequent damage category.
* Potholes are the least represented class.
* A significant proportion of damage instances occupy a very small fraction of the image area.
* Small object prevalence motivates the use of SAHI-based sliced inference.

---

## Baseline Detection Experiments

Two baseline object detection models were trained using identical hyperparameters.

### YOLOv8n Baseline

Training Configuration:

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

Training Configuration:

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
* The largest improvements were observed in precision and mAP scores.
* Pothole detection remained the most challenging category.
* Other Corruption achieved the highest overall detection accuracy.
* Small object detection remained a significant challenge.

---

## SAHI + YOLOv8n Experiment

To improve small-object detection performance, SAHI (Slicing Aided Hyper Inference) was integrated with the trained YOLOv8n detector.

### SAHI Configuration

| Parameter            | Value |
| -------------------- | ----- |
| Slice Height         | 640   |
| Slice Width          | 640   |
| Overlap Height Ratio | 0.20  |
| Overlap Width Ratio  | 0.20  |
| Confidence Threshold | 0.25  |

### Quantitative Evaluation

A subset of 2000 validation images was evaluated using both standard YOLOv8n inference and SAHI-assisted inference.

| Method         | Total Detections |
| -------------- | ---------------: |
| YOLOv8n        |             3773 |
| YOLOv8n + SAHI |             4564 |

### Detection Improvement

```text
YOLOv8n detections: 3773
SAHI detections: 4564
Improvement: 20.96%
```

### Observations

* SAHI detected substantially more road damage instances than standard inference.
* Improvements were most noticeable for small and distant defects.
* SAHI reduced missed detections in cluttered road scenes.
* Qualitative comparison figures demonstrated improved sensitivity toward small cracks and potholes.
* Increased detections came at the cost of higher inference time.

---

## Visualizations Generated

The project includes:

### Prediction Visualizations

* YOLOv8n predictions
* YOLOv8s predictions
* SAHI-enhanced predictions

### Comparative Figures

* YOLOv8n vs YOLOv8n + SAHI
* Top-gain comparison examples
* Qualitative detection analysis

### Thesis Figures

* Sample detections
* Class distribution plots
* Bounding box statistics
* Detection comparison figures

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
* SAHI + YOLOv8n Evaluation
* Quantitative Detection Comparison
* Repository Organization
* Experimental Result Documentation

### In Progress

* SAHI + YOLOv8s Evaluation

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

1. SAHI + YOLOv8s Evaluation
2. YOLOv8n vs YOLOv8s SAHI Comparison
3. Crack Segmentation using U-Net
4. Crack500 Dataset Integration
5. DeepCrack Dataset Evaluation
6. Detection–Segmentation Pipeline Integration
7. Final Thesis Documentation
8. Research Paper Preparation

---

## Thesis Goal

To develop an efficient deep learning framework capable of accurately detecting and segmenting road damage, with a particular focus on improving the detection of small cracks and potholes through SAHI-based sliced inference and pixel-level segmentation using U-Net.

The final system aims to support automated road condition monitoring and intelligent infrastructure maintenance.

---

## Author

**Aadeesh Ranjan**

B.Tech Computer Science and Engineering

Road Damage Detection and Segmentation Thesis Project
