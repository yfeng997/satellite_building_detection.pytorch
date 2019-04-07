import glob
import random
import os
import numpy as np
import json
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

import sys

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob('%s/*.*' % folder_path))
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image
        img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=224):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'annotations').replace('.png', '.json').replace('.jpg', '.json') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()

        #---------
        #  Label
        #---------

        # raw_image = cv2.imread(img_path)

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                # [{class: class, bbox: [xywh]}]
                raw_labels = json.load(f)
            for raw_label in raw_labels:
                class_label = int(raw_label['class'])
                bbox = raw_label['bbox']
                # ASSUME we do not need to pad
                x_c = bbox[0] + bbox[2]/2
                y_c = bbox[1] + bbox[3]/2
                w = bbox[2]
                h = bbox[3]
                labels.append([class_label, x_c, y_c, w, h])
                
                # x = int(bbox[0] * 224)
                # y = int(bbox[1] * 224)
                # w = int(bbox[2] * 224)
                # h = int(bbox[3] * 224)
                # if class_label == 0:
                #     cv2.rectangle(raw_image, (x, y), (x+w, y+h), (255,0,0), 1)
                # else:
                #     cv2.rectangle(raw_image, (x, y), (x+w, y+h), (0,255,0), 1)
            # labels = np.loadtxt(label_path).reshape(-1, 5)
            # # Extract coordinates for unpadded + unscaled image
            # x1 = w * (labels[:, 1] - labels[:, 3]/2)
            # y1 = h * (labels[:, 2] - labels[:, 4]/2)
            # x2 = w * (labels[:, 1] + labels[:, 3]/2)
            # y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # # Adjust for added padding
            # x1 += pad[1][0]
            # y1 += pad[0][0]
            # x2 += pad[1][0]
            # y2 += pad[0][0]
            # # Calculate ratios from coordinates
            # labels[:, 1] = ((x1 + x2) / 2) / padded_w
            # labels[:, 2] = ((y1 + y2) / 2) / padded_h
            # labels[:, 3] *= w / padded_w
            # labels[:, 4] *= h / padded_h
            # imagepathh = img_path.split('/')[-1]
            # cv2.imwrite('/home/yuansong/code/building-detection/tmp/'+imagepathh, raw_image)
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if not len(labels) == 0:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)
