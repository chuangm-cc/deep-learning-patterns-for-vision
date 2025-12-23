# deep-learning-patterns-for-vision

A collection of practical computer vision projects implemented in **PyTorch**, covering **image classification**, **face detection**, **semantic segmentation**, and **GAN-based image generation**, along with a **real-time assistive perception system** that integrates object detection and depth estimation.

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
Implemented full training and evaluation pi
