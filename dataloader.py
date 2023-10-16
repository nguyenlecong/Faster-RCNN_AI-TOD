import os

import torch
from PIL import Image

from modules import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # Follow Fast R-CNN paper
    return T.Compose(transforms)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.label_root = root.replace('/hdd/thaihq/qnet_search/ori_data', '/hdd/nguyenlc/ai_training/faster_rcnn/qnet_search/correct_labels')
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        # self.labels = list(sorted(os.listdir(os.path.join(self.label_root, "labels"))))
        self.labels = list(sorted(os.listdir(self.label_root)))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.label_root, self.labels[idx])
        
        img = Image.open(img_path).convert("RGB")
        with open(label_path) as file:
            label = [line.rstrip() for line in file]

        num_objs = len(label)
        boxes = []
        labels = []
        for i in range(num_objs):
            # xmin, ymin, xmax, ymax, cls = [int(j) for j in label[i].split(', ')]
            # boxes.append([xmin, ymin, xmax, ymax])
            xmin, ymin, width, height, cls = [int(j) for j in label[i].split(', ')]
            boxes.append([xmin, ymin, xmin+width, ymin+height])
            labels.append(cls-1)  #**********

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_name"] = os.path.basename(img_path)[:-4]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)