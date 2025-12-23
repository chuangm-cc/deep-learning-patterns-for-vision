# Face Detection with Deep Learning

This project implements a complete anchor-based face detection system, covering data analysis, model architecture design, training, and inference.  
**For detailed implementation, experiments, and analysis, please refer to the full report:**  
📄 [`face_detection_report.pdf`](./face_detection_report.pdf)

## Overview

This project builds a modern face detector following principles used in real-world detection systems.  
The design is closely related to **region-based detection pipelines (e.g., Fast R-CNN)**, while adopting a **fully convolutional, multi-scale architecture** for efficiency.

The implementation emphasizes how region-level reasoning, feature pyramids, and anchor-based matching are combined in practice.

## Detection Pipeline

The detector follows a region-based detection paradigm:

- Convolutional backbone extracts shared feature maps (similar to Fast R-CNN shared computation)
- Multi-scale feature pyramids enable detection of faces at different resolutions
- Anchors act as dense region proposals
- Region-wise classification and regression refine candidate face locations
- Post-processing with Non-Maximum Suppression (NMS) produces final detections

This design bridges classical **Fast R-CNN concepts** with modern fully convolutional detection frameworks.

## Model Architecture

The detector consists of the following components:

- **ResNet-50 backbone** for feature extraction  
- **Feature Pyramid Network (FPN)** for multi-scale feature aggregation  
- **SSH modules** for enhanced contextual representation  
- **Multi-head prediction layers** for:
  - Binary face / background classification  
  - Bounding box regression  
  - Facial landmark regression  

## Results

- Robust face detection across scales and crowded scenes
- Accurate bounding box localization and landmark estimation
- Stable multi-task training behavior
- Qualitative detection results on real-world images
