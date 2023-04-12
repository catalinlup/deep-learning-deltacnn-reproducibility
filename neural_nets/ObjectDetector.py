import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


class ObjectDetector(nn.Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
        self.feature_extractor = baseModel.features
        self.feature_extractor.out_channels = baseModel.last_channel

        self.anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
        
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
        
        self.faster_rcnn = FasterRCNN(self.feature_extractor, numClasses, rpn_anchor_generator=self.anchor_generator, box_roi_pool=self.roi_pooler)

    def forward(self, x, targets=None):
       return self.faster_rcnn(x, targets)
    

    def train(self):
        """
        Switches the model to train mode
        """
        return self.faster_rcnn.train()

    def eval(self):
        """
        Switches the model to evaluation mode
        """
        return self.faster_rcnn.eval()