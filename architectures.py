from neural_nets.mobilenet_original import mobilenet_v2
from neural_nets.mobilenet_deltacnn import DeltaCNN_mobilenet_v2
from neural_nets.ObjectDetector import ObjectDetector
from config import *

ARCHS = {
    'mobilenet_original': lambda: ObjectDetector(mobilenet_v2(pretrained=True).to(DEVICE), NUM_CLASSES).to(DEVICE),
    'mobilenet_deltacnn': lambda: ObjectDetector(DeltaCNN_mobilenet_v2(pretrained=True).to(DEVICE), NUM_CLASSES, deltacnn=True).to(DEVICE)
}