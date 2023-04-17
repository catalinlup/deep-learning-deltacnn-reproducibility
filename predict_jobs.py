from data_loaders.penn_fudan import PennFudanDataset
import torch
from config import *
from helpers import get_transform, collate_fn, show_image, get_device



def get_pennfudan():
    dataset = PennFudanDataset('data/PennFudanPed/PennFudanPed', transforms=get_transform(True, size=IMG_SIZE))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=collate_fn)
    
    return data_loader


PREDICT_JOBS = {
    'classic': {
        'model_name': 'mobilenet_classic',
        'architecture_name': 'mobilenet_original',
        'data_loader': get_pennfudan
    }
}