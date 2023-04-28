import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
from config import IMG_SIZE


class Mot16Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, sequence_name, original_width=1960, original_height=1080):
        self.root = root
        self.transforms = transforms

        self.train_test_path = 'train'
        self.sequence_name = sequence_name

        self.original_width = original_width
        self.original_height = original_height

        self.imgs = list(sorted(os.listdir(os.path.join(root, self.train_test_path, self.sequence_name, 'img1'))))
        self.annotations_path =  os.path.join(root, self.train_test_path, self.sequence_name, 'gt', 'gt.txt')
        self.annotations = pd.read_csv(self.annotations_path, header=None, usecols=[0, 2, 3, 4, 5], names=['id', 'x', 'y', 'w', 'h'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.train_test_path, self.sequence_name, 'img1', self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        frame_id = idx + 1

        relevant_annotations = self.annotations[self.annotations.id == frame_id]
        # print(relevant_annotations)
        boxes = torch.tensor(relevant_annotations[['x', 'y', 'w', 'h']].values, dtype=torch.float32)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        boxes[:, 0] /= self.original_width
        boxes[:, 2] /= self.original_width
        boxes[:, 1] /= self.original_height
        boxes[:, 3] /= self.original_height

        boxes[:, 0] *= IMG_SIZE[0]
        boxes[:, 2] *= IMG_SIZE[0]
        boxes[:, 1] *= IMG_SIZE[1]
        boxes[:, 3] *= IMG_SIZE[1]

        num_objs = boxes.shape[0]
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([frame_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        scores = torch.ones((num_objs,), dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['scores'] = scores

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
    def __len__(self):
        return len(self.imgs)
