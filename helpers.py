import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt

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


def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'