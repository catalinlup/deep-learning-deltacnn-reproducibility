from data_loaders.penn_fudan import PennFudanDataset
from helpers import get_transform, collate_fn, show_image, get_device
import torch
import torch.nn
from neural_nets.mobilenet_original import mobilenet_v2
from neural_nets.ObjectDetector import ObjectDetector
from config import *
import numpy as np
from architectures import ARCHS
from train_jobs import TRAIN_JOBS
import sys
import matplotlib.pyplot as plt
import cv2


train_job = TRAIN_JOBS[sys.argv[1]]

if len(sys.argv) > 2:
    OUTPUT_PLOT = bool(sys.argv[2])
else:
    OUTPUT_PLOT = False


dataset = PennFudanDataset('data/PennFudanPed/PennFudanPed', transforms=get_transform(True, size=IMG_SIZE))

data_loader = train_job['data_loader']()

# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=1, shuffle=False, num_workers=4,
#     collate_fn=collate_fn)


model = ARCHS[train_job['architecture_name']]()
model.train()



# Constructing an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=train_job['lr'], weight_decay=train_job['weight_decay'])
# and a learning rate scheduler

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Trains the model for one epoch
    """


    all_losses = []

    total = len(data_loader)

    for i, (img, target) in enumerate(data_loader):

        optimizer.zero_grad()

        input = torch.stack(img).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in target]

        loss_dict = model(input, targets)
        # print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()

        # handle the losses
        loss_value = losses.item()
        all_losses.append(loss_value)
        if i % print_freq == 0:
            print(f'Epoch {epoch} - {round(i / total) * 100}% - Loss: {loss_value}')

    return np.mean(all_losses)
    

epoch_losses = []
for i in range(train_job['epochs']):
    epoch_loss = train_one_epoch(model, optimizer, data_loader, DEVICE, i, 2)
    print('Epoch loss ', epoch_loss)
    epoch_losses.append(epoch_loss)



if OUTPUT_PLOT:
    plt.plot(epoch_losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train Loss')

    plt.savefig(f'./saved_plots/{train_job["model_name"]}.png')

torch.save(model.state_dict(), f'./saved_models/{train_job["model_name"]}.pt')