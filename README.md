# deep-learning-patterns-for-vision

A collection of practical computer vision projects implemented in **PyTorch**, covering **face detection**, **semantic segmentation**, **GAN-based image generation**, and **image classification**, along with a **real-time assistive perception system** that integrates object detection and depth estimation.

Detailed design choices and results are documented in the reports inside each project folder.

---

## Face Detection: Anchor-based Detector  
- Folder: [`face-detection`](./face-detection)

Implemented a complete **anchor-based face detection system** using a **ResNet-50 backbone**, **Feature Pyramid Network (FPN)**, and **SSH modules**.  
The pipeline includes multi-head prediction for **classification, bounding box regression, and facial landmarks**, with **IoU-based anchor matching** and **Non-Maximum Suppression (NMS)** for post-processing.

**Tech:** PyTorch, ResNet, FPN, Anchor-based Detection, IoU, NMS

---

## Semantic Segmentation: DeepLabV3 / DeepLabV3+  
- Folder: [`semantic-segmentation`](./semantic-segmentation)

Built **DeepLabV3 and DeepLabV3+** semantic segmentation models from core components, including **ASPP modules** and DeepLab heads on ResNet backbones.  
Implemented full training and evaluation pipelines using **mIoU**, with attention to **class imbalance** in pixel-level prediction.

**Tech:** PyTorch, DeepLab, ASPP, Semantic Segmentation, mIoU

---

## GAN Image Generation: DCGAN  
- Folder: [`gan-generation`](./GAN-generation)

Implemented **DCGAN** with **upsample + convolution** generator blocks and convolutional discriminators.  
Applied architectural refinements to improve training stability and image generation quality.

**Tech:** PyTorch, GAN, DCGAN, Convolutional Generative Models

---

## Image Classification: MNIST / CIFAR-100  
- Folder: [`image-classification`](./image-classification)

Trained **FCN and CNN** models for image classification on **MNIST** and **CIFAR-100**, including custom loss implementation, model tuning, and feature analysis.  
Visualized learned representations using **t-SNE** to study class separability.

- Best CIFAR-100 accuracy: **~61%**

**Tech:** PyTorch, CNN, Image Classification, t-SNE

---

## FollowMe: Assistive Perception  
- Folder: [`followme-assistive-perception`](./FollowMe-Blind-Aid)

A real-time indoor assistive perception system for obstacle awareness.  
The system combines **YOLOv5 object detection** with **MiDaS-based depth estimation**, fusing RGB and depth information to estimate obstacle distance and provide **low-latency audio feedback**.  
Designed lightweight post-processing for improved robustness in real-world hallway environments.

- **~85% system accuracy**
- **~120ms average latency** (worst-case ~230ms)

**Tech:** YOLOv5, MiDaS, Depth Estimation, Real-time Vision, Multi-sensor Fusion

---

**Author:** Chuang Ma
