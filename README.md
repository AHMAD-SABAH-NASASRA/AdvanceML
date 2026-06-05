# Drone Person & Car Detection

Final Project for Advance Machine Learning

## Team Members

- Mohammad Ismail
- Ahmad Nasasra

---

# Project Overview

This project focuses on detecting and localizing **persons** and **cars** in aerial drone imagery using the **YOLOv10** object detection framework and the **VisDrone2019** dataset.

Drone-based object detection presents unique challenges due to:

- Small object sizes
- Crowded scenes
- Occlusions
- Varying camera altitudes and viewpoints
- Complex urban environments

The goal of this project is to build an accurate detector capable of identifying persons and vehicles in real-world drone images.

---

# Dataset

Dataset used:

**VisDrone2019 Object Detection Dataset**

The dataset contains thousands of aerial images collected by drones in different environments and weather conditions.

Original dataset classes were filtered and reduced to:

| Class ID | Class |
|-----------|---------|
| 0 | Person |
| 1 | Car |

---

# Project Pipeline

## 1. Data Preparation

- Converted VisDrone annotations into YOLO format
- Filtered target classes (Person and Car)
- Generated YOLO dataset configuration files
- Organized train, validation, and test splits

---

## 2. Data Visualization

Before training, samples and annotations were visualized to verify:

- Bounding box correctness
- Label conversion accuracy
- Class distribution

---

## 3. Data Augmentation

The following augmentations were used during training:

- Mosaic Augmentation
- MixUp
- Random Scaling
- Translation
- Horizontal Flipping

These augmentations improved generalization and helped detect small objects.

---

## 4. Model Selection

Model:

**YOLOv10x**

Reasons:

- Strong detection performance
- Efficient training
- Excellent small-object detection capabilities
- Suitable for aerial imagery

---

# Training Configuration

| Parameter | Value |
|------------|--------|
| Model | YOLOv10x |
| Image Size | 1280 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.0008 |
| Epochs | 100 |
| Early Stopping | 25 |
| Batch Size | Auto |
| Mosaic | Enabled |
| MixUp | Enabled |
| Scale Augmentation | Enabled |
| Translation | Enabled |

---

# Experiments

## Baseline Model

Initial training configuration:

- Smaller model
- Lower image resolution

Results:

| Metric | Score |
|----------|---------|
| mAP50 | 0.544 |
| mAP50-95 | 0.310 |

---

## Final Optimized Model

Improvements:

- YOLOv10x
- Image Size = 1280
- Improved augmentation
- Longer training schedule

Results:

| Metric | Score |
|----------|---------|
| Precision | 0.807 |
| Recall | 0.742 |
| mAP50 | 0.807 |
| mAP50-95 | 0.521 |

---

# Performance Improvement

| Metric | Baseline | Final |
|----------|-----------|--------|
| mAP50 | 0.544 | 0.807 |
| mAP50-95 | 0.310 | 0.521 |

Significant improvement was achieved through model scaling, larger image resolution, and optimized training settings.

---

# Training Analysis

The model converged successfully during training.

Observations:

- Rapid improvement during early epochs
- Stable convergence after approximately epoch 60
- Best performance achieved around epoch 67
- Additional epochs produced minimal improvement
- Early stopping prevented unnecessary training

---

# Results

Final validation performance:

| Metric | Value |
|----------|---------|
| Precision | 0.807 |
| Recall | 0.742 |
| mAP50 | 0.807 |
| mAP50-95 | 0.521 |

The detector successfully identified:

- Pedestrians
- Cars

across challenging aerial scenes.

---

# Example Predictions

Examples of model predictions are available in:

```text
images/
```

Including:

- Detection examples
- Precision-Recall curves
- Confusion matrices
- Training curves

---

# Project Structure

```text
AdvanceML/
│
├── train.py
├── evaluate.py
├── inference.py
├── visualization.py
├── dataset_utils.py
├── config.py
│
├── dataset/
│
├── images/
│   ├── results.png
│   ├── confusion_matrix_normalized.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   ├── prediction1.jpg
│   ├── prediction2.jpg
│   └── prediction3.jpg
│
├── visdrone-yolov10-object-detection.py
├── visdrone-yolov10-object-detection.ipynb
│
└── README.md
```

---

# Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv10
- OpenCV
- NumPy
- Matplotlib

---

# Challenges

The main challenges encountered were:

- Detecting very small pedestrians
- Dense urban scenes
- Class imbalance
- Computational cost of high-resolution training

---

# Future Work

Potential improvements include:

- Multi-class detection
- Model quantization
- Knowledge distillation
- Real-time deployment on edge devices
- Tracking and trajectory analysis

---

# Course Information

**Course:** Advance Machine Learning

**Year:** 2026

---

# Acknowledgment

Dataset provided by the VisDrone Challenge.

YOLO implementation provided by Ultralytics.
