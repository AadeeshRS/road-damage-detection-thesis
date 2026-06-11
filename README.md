# Road Damage Detection and Segmentation using YOLOv8, SAHI, and U-Net

## Overview

This B.Tech thesis project focuses on the automated detection and segmentation of road damage using deep learning techniques. The proposed framework combines YOLOv8 for object detection, SAHI (Slicing Aided Hyper Inference) for improved small-object detection, and U-Net for pixel-level segmentation of road defects.

The objective is to improve the detection and localization of road damage such as cracks and potholes from road-view images captured using vehicle-mounted cameras.

---

## Problem Statement

Road damage such as cracks and potholes can significantly affect transportation safety and infrastructure maintenance. Manual inspection methods are labor-intensive, time-consuming, and difficult to scale. Deep learning-based approaches provide an opportunity to automate road condition monitoring; however, detecting small and irregular road defects remains challenging due to variations in road textures, lighting conditions, shadows, weather conditions, and the limited size of damage regions within an image.

This project aims to develop an automated road damage detection and segmentation pipeline using YOLOv8, SAHI, and U-Net.

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

* RDD2022 Road Damage Dataset

### Segmentation Dataset

* Crack500 (Planned)
* DeepCrack (Alternative)

---

## Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* SAHI
* OpenCV
* Segmentation Models PyTorch
* Google Colab
* VS Code
* GitHub

---

## Repository Structure

```text
road-damage-detection-thesis/
│
├── datasets/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── detection/
│
├── sahi/
│
├── segmentation/
│
├── experiments/
│
├── reports/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Current Progress

### Completed

* Literature Review
* Research Paper Analysis
* Project Structure Setup
* Dataset Acquisition (RDD2022)

### In Progress

* Dataset Exploration

### Upcoming

* YOLOv8 Baseline Training
* SAHI Integration
* U-Net Segmentation
* Detection and Segmentation Pipeline Integration
* Experimental Evaluation
* Thesis Documentation

---

## Evaluation Metrics

### Detection

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95

### Segmentation

* Dice Score
* Intersection over Union (IoU)
* Pixel Accuracy

---

## Thesis Goal

To develop an efficient deep learning framework capable of accurately detecting and segmenting road damage, with a particular focus on improving the detection of small cracks and potholes through SAHI-based sliced inference.

---
