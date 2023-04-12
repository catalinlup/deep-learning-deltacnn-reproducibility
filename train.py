from data_loaders.penn_fudan import PennFudanDataset
from helpers import get_transform, collate_fn, show_image, get_device
import torch
import torch.nn
from neural_nets.mobilenet_original import mobilenet_v2
from neural_nets.ObjectDetector import ObjectDetector
from config import *
import numpy as np


dataset = PennFudanDataset('data/PennFudanPed/PennFudanPed', transforms=get_transform(True, size=IMG_SIZE))

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    collate_fn=collate_fn)

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=1, shuffle=False, num_workers=4,
#     collate_fn=collate_fn)


backbone = mobilenet_v2(pretrained=True).to(DEVICE)
object_detector = ObjectDetector(backbone, 2).to(DEVICE)


# Constructing an optimizer
params = [p for p in object_detector.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)
# and a learning rate scheduler

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trains the model for one epoch
    """

    model.train()

    all_losses = []

    total = len(data_loader)

    for i, (img, target) in enumerate(data_loader):

        optimizer.zero_grad()

        input = torch.stack(img).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]

        loss_dict = model(input, targets)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        # handle the losses
        loss_value = losses.item()
        all_losses.append(loss_value)
        if i % print_freq == 0:
            print(f'Epoch {epoch} - {i / total * 100}% - Loss: {loss_value}')

    return np.mean(all_losses)
    


print(train_one_epoch(object_detector, optimizer, data_loader, DEVICE, 0, 10))