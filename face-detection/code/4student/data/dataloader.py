import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class mydataloader(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        # TODO
        file = open(txt_path + "label.txt", 'r')
        lines = file.readlines()

        word = []
        for l in lines:
            # get to next image
            if (l[0] == "#"):
                if(len(word)!=0):
                    self.words.append(word)
                    # print(word)
                img_name = l.split(' ')
                img_path = txt_path + 'images/' + img_name[1][:-1]
                # print(img_path)
                self.imgs_path.append(img_path)

                word = []

                # print(img_path)

            else:
                lang_split = l.split(" ")
                float_e = [float(element) for element in lang_split]
                word.append(float_e)
                # print(float_e)



    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            # TODO
            annotation = np.zeros((1,15))
            x, y, w, h = label[:4]
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            annotation[0][0] = x
            annotation[0][1] = y
            annotation[0][2] = x+w
            annotation[0][3] = y+h

            land_xs = label[4:-1:3]
            land_ys = label[5:-1:3]
            for i in range(len(land_xs)):
                # if (float(land_xs[i]) > 0):
                annotation[0][4+2*i] = float(land_xs[i])
                # if (float(land_ys[i]) > 0):
                annotation[0][5+i*2] = float(land_ys[i])
            if(label[4]<0):
                annotation[0][14] = -1
            else:
                annotation[0][14]=1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

# m = mydataloader('../../widerface_homework/train/')
# print(m[0])
# print(m[2])