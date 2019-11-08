import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import pdb

class BuildingDetectionDataset(Dataset):
    """Dataset for loading images and annotations from a config file. 
    Each line contains information for both the image and annotation. 
    E.g., '/data/feng/building-detect/images/1225.jpg' is the image path, 
    and the corresponding annotation path is '/data/feng/building-detect/annotations/1225.json'

    The class is designed specifically to work with torchvision.models.detection.faster_rcnn.
    Check the output from __getitem__ to understand the specific data format. 
    """
    def __init__(self, config_file, transform=transforms.ToTensor()):
        """
        Args:
            config_file (string): path to config file of all data, each line 
                consists of an image path
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(config_file, 'r') as file:
            self.img_paths = file.readlines()
        self.label_paths = [path.replace('images', 'annotations').replace('.jpg', '.json') for path in self.img_paths]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx].rstrip()
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        
        # Load annotation
        label_path = self.label_paths[idx].rstrip()
        with open(label_path, 'r') as f:
            # [{class: 0/1, bbox: [x,y,w,h]}, {class, bbox}, ...]
            raw_labels = json.load(f)
        boxes = []
        labels = []
        channel, img_w, img_h = img.shape
        for raw_label in raw_labels:
            # [x, y, w, h], normalized
            bbox = raw_label['bbox']
            x1 = bbox[0]*img_w
            y1 = bbox[1]*img_h
            x2 = (bbox[0]+bbox[2])*img_w
            y2 = (bbox[1]+bbox[3])*img_h
            # [x1, y1, x2, y2], image-sized
            boxes.append([x1, y1, x2, y2])
            labels.append(raw_label['class'])

        target = {
            "boxes": torch.FloatTensor(boxes), 
            "labels": torch.LongTensor(labels), 
            "area": torch.LongTensor(0),
            "iscrowd": torch.LongTensor(0)
        }

        return img, target

def custom_collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return (data, target)

# dataset = BuildingDetectionDataset('config/train_config.txt')
# for (img, target) in dataset:
#     pdb.set_trace()

