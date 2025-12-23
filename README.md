# deep-learning-patterns-for-vision

A collection of practical computer vision projects implemented in **PyTorch**, spanning **image classification**, **face detection**, **semantic segmentation**, and **GAN-based image generation**, plus a **real-time assistive perception system** that integrates detection and depth estimation for obstacle awareness.

Detailed technical explanations, experiments, and ablation studies can be found in the corresponding reports inside each folder.

---

## Face Detection: Anchor-based Detector (Backbone + FPN + Multi-head)

Implemented a full **anchor-based face detection pipeline** with a **ResNet-50 backbone**, **Feature Pyramid Network (FPN)**, and **SSH modules**.  
The detector includes classification, bounding box regression, and facial landmark heads, along with **IoU-based anchor matching**, **multi-task loss**, and **Non-Maximum Suppression (NMS)**.

Key components:
- Custom dataset loading and statistical analysis
- Anchor generation and IoU matching strategy
- Multi-head detection (class / bbox / landmarks)
- End-to-end training and qualitative visualizations

- Folder: [`face-detection`](./face-detection)

---

## Semantic Segmentation: DeepLabV3 / DeepLabV3+ from Core Blocks

Built **DeepLabV3 and DeepLabV3+** style semantic segmentation networks from core components, including **ASPP modules** and DeepLab heads on top of ResNet backbones.

Highlights:
- Manual implementation of ASPP and DeepLab heads
- Training pipeline with **backbone learning rate scaling**
- Step learning rate scheduler
- Evaluation using **mIoU**
- Experiments with alternative loss functions for **class imbalance**

- Folder: [`semantic-segmentation`](./semantic-segmentation)

---

## GAN Image Generation: DCGAN + Training Stabilization

Implemented **DCGAN** for image generation using **upsample + convolution** blocks in the generator.  
Explored architectural and training strategies to improve convergence and visual quality.

Experiments include:
- Data augmentation for discriminator regularization
- Residual blocks in generator and discriminator
- Training stability analysis and loss visualization

- Folder: [`gan-generation`](./GAN-generation)

---

## Image Classification: MNIST / CIFAR-100 + Fine-tuning + t-SNE

Trained and analyzed **Fully Connected Networks (FCN)** and **CNNs** for image classification on **MNIST** and **CIFAR-100**.

Includes:
- From-scratch loss function implementation
- Architecture and hyperparameter tuning
- Training and evaluation curves
- **t-SNE visualization** of learned feature embeddings

- Best CIFAR-100 accuracy: **~61%**
- Folder: [`image-classification`](./image-classification)

---

## FollowMe: Assistive Perception (Detection + Depth + Audio)

**FollowMe** is a real-time indoor assistive perception prototype designed for
