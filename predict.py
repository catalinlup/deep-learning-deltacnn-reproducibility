from helpers import get_transform, collate_fn, show_image_cv2, get_device, compute_iou
import torch
import torch.nn
from neural_nets.mobilenet_original import mobilenet_v2
from neural_nets.ObjectDetector import ObjectDetector
from config import *
import numpy as np
from architectures import ARCHS
from predict_jobs import PREDICT_JOBS
import sys
import matplotlib.pyplot as plt
import cv2
import time
import json


predict_job = PREDICT_JOBS[sys.argv[1]]

if len(sys.argv) > 2:
    OUTPUT_IMAGE = bool(sys.argv[2])
else:
    OUTPUT_IMAGE = False

model = ARCHS[predict_job['architecture_name']]()
model.load_state_dict(torch.load(f"./saved_models/{predict_job['model_name']}.pt"))
model.eval()


data_loader = predict_job['data_loader']()

prediction_times = []
ious = []

for i, (img, target) in enumerate(data_loader):

    time0 = time.time()

    input = torch.stack(img).to(DEVICE)
    predictions = model(input)


    time1 = time.time()

    prediction_times.append(time1 - time0)
    ious.append(compute_iou(predictions[0], target[0]))
    
    if OUTPUT_IMAGE:
        show_image_cv2(img[0], predictions[0])
        cv2.waitKey(1)


avg_prediction_time = np.mean(prediction_times)
avg_fps = 1.0 / avg_prediction_time
avg_iou = np.mean(ious)

with open(f"./saved_results/{predict_job['model_name']}.json", 'w') as file:
    results = {
        'avg_prediction_time': avg_prediction_time,
        'avg_fps': avg_fps,
        'avg_iou': avg_iou
    }
    json.dump(results, file)
    

if OUTPUT_IMAGE:
    cv2.waitKey(0)
    cv2.destroyAllWindows()

