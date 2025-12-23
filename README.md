# deep-learning-patterns-for-vision

A collection of practical computer vision projects in PyTorch, spanning **image classification**, **face detection**, **semantic segmentation**, and **GAN-based image generation**, plus a **real-time assistive perception system** that integrates detection + depth estimation for obstacle awareness.

### Face Detection: Anchor-based Detector (Backbone + FPN + Multi-head)
Implemented an anchor-based face detector with **ResNet50 backbone**, **FPN**, **SSH modules**, classification/regression/landmark heads, plus IoU-based matching and NMS. Includes dataset statistics and qualitative visualizations.
- Folder: [`face-detection`](./face-detection)

### Semantic Segmentation: DeepLabV3/V3+ from Core Blocks
Built DeepLab-style segmentation networks (ASPP modules, DeepLab heads), training loop with backbone LR scaling and step LR scheduler, and experimented with alternative loss functions for class imbalance.
- Folder: [`semantic-segmentation`](./semantic-segmentation)

### GAN Image Generation: DCGAN + Training Stabilization
Implemented DCGAN with **upsample+conv** generator blocks, added data augmentation, and explored architectural variants (e.g., residual blocks) to improve convergence and visual quality.
- Folder: [`gan-generation`](.GAN-generation)

### Image Classification: MNIST / CIFAR-100 + Fine-tuning + t-SNE
Trained FCN/CNN baselines and ran systematic fine-tuning on CIFAR-100. Includes training curves, prediction visualizations, and t-SNE feature embeddings.
- Best CIFAR-100 accuracy: **~61%**
- Folder: [`image-classification`](.image-classification)

### FollowMe: Assistive Perception (Detection + Depth + Audio)
A real-time indoor hallway assistance prototype on embedded hardware, combining **YOLOv5 (pretrained + self-trained)** with **MiDaS-based relative depth** to make distance estimation more robust under noisy depth measurements. Includes lightweight post-processing (whitelisting / occurrence counting / pixel selection) and audio feedback.
- Key results: **~85% system accuracy**, **~120ms avg latency** (worst-case ~230ms)
- Folder: [`followme-assistive-perception`](./followme-assistive-perception)

--- 
Author： Chuang Ma

