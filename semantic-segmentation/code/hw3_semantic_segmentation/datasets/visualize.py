import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datetime



def visualize_20(root,img_names):
  current_datetime = datetime.datetime.now()
  timestamp_str = current_datetime.strftime('%Y-%m-%d')
  owner ='chuangm'
  base_dir = "VOCdevkit/VOC2012"
  voc_root = os.path.join(root, base_dir)
  fig, axes = plt.subplots(10,4, figsize=(10, 20))

  image_dir = os.path.join(voc_root, 'JPEGImages')
  annotation_dir = os.path.join(voc_root, 'SegmentationClass')
  # Loop to load and plot images and annotations
  for i in range(10):
      for j in range(2):
          idx=i*2+j
          img=os.path.join(image_dir, img_names[idx]+'.jpg')
          annotation =os.path.join(annotation_dir,img_names[idx]+'.png')
          img = Image.open(img)
          annotation = Image.open(annotation)
          axes[i, j*2].imshow(img)
          axes[i, j*2+1].imshow(annotation)
          axes[i, j*2].axis('off')
          axes[i, j*2+1].axis('off')
  plt.suptitle(f'visualization of 20 images\n {owner}: {timestamp_str}', fontsize=16)
  plt.savefig("visualization_res.png")


images = ["2007_000250","2007_000733","2007_001420","2007_003178","2007_004830",
  "2007_000032","2007_000039","2007_000061","2007_000063","2007_000068",
  "2007_000175","2007_000452","2007_000491","2007_000528","2007_000648",
  "2007_002227","2007_002284","2007_003251","2007_003367","2007_000793",
  ]
visualize_20('data',images)