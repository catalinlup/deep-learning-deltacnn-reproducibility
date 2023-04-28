import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision.ops import box_iou

def get_transform(train, size=(128, 128)):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    transforms.append(T.Resize(size, antialias=True))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def collate_fn(batch):
    return tuple(zip(*batch))


def show_image(img):
    """
    Displays an image
    """
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

def show_image_cv2(img, predictions):
    image = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    boxes = predictions['boxes']

    scores = predictions['scores']

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
        
        cv2.rectangle(image,(x1, y1),(x2, y2),(0,255,0),2)
        cv2.putText(image,f'{score * 199}',(x2+10, y2), 0,0.3,(0,255,0))
    cv2.imshow('Detection stream', image)

    
def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

def compute_iou(prediction, target):
    """
    Computes the iou of the object detector.
    """

    target_boxes = target['boxes'].to(get_device())
    predicted_boxes = prediction['boxes'].to(get_device())

    ious = box_iou(target_boxes, predicted_boxes)

    max_ious = torch.max(ious, dim=1)[0].reshape(len(target_boxes))

    return max_ious.mean().item()




