from data_loaders.penn_fudan import PennFudanDataset
from data_loaders.mot16 import Mot16Dataset
import torch
from config import *
from helpers import get_transform, collate_fn, show_image, get_device

def get_pennfudan():
    dataset = PennFudanDataset('data/PennFudanPed/PennFudanPed', transforms=get_transform(False, size=IMG_SIZE))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)
    
    return data_loader

def get_mot16():
    dataset = Mot16Dataset('data/MOT16', transforms=get_transform(False, size=IMG_SIZE), sequence_name='MOT16-04')

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)
    
    return data_loader


TRAIN_JOBS = {
    'classic': {
        'model_name': 'mobilenet_classic',
        'architecture_name': 'mobilenet_original',
        'lr': 0.001,
        'weight_decay': 0.0005,
        'epochs': 2,
        'data_loader': get_pennfudan

    },

    'delta_cnn': {
        'model_name': 'mobilenet_deltacnn',
        'architecture_name': 'mobilenet_classic',
        'lr': 0.001,
        'weight_decay': 0.0005,
        'epochs': 10,
        'data_loader': get_pennfudan
    }
}