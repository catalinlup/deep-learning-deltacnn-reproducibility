from data_loaders.penn_fudan import PennFudanDataset
from data_loaders.mot16 import Mot16Dataset
import torch
from config import *
from helpers import get_transform, collate_fn, show_image, get_device



def get_pennfudan():
    dataset = PennFudanDataset('data/PennFudanPed/PennFudanPed', transforms=get_transform(False, size=IMG_SIZE))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)
    
    return data_loader

def get_mot16():
    dataset = Mot16Dataset('data/MOT16', transforms=get_transform(False, size=IMG_SIZE), sequence_name='MOT16-02')

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)
    
    return data_loader


PREDICT_JOBS = {
    'classic': {
        'model_name': 'mobilenet_classic',
        'architecture_name': 'mobilenet_original',
        'data_loader': get_pennfudan
    },
    
    'classic_lr3_wd4': {
        'model_name': 'mobilenet_classic_lr3_wd4',
        'architecture_name': 'mobilenet_original',
        'data_loader': get_mot16

    },
    
    'classic_lr3_wdb1': {
        'model_name': 'mobilenet_classic_lr3_wdb1',
        'architecture_name': 'mobilenet_original',
        'data_loader': get_mot16

    },
    'classic_lr3_wd0': {
        'model_name': 'mobilenet_classic_lr3_wd0',
        'architecture_name': 'mobilenet_original',
        'data_loader': get_mot16

    },
    
    'classic_lr3_wd015': {
        'model_name': 'mobilenet_classic_lr3_wd015',
        'architecture_name': 'mobilenet_original',
        'data_loader': get_mot16

    },
}