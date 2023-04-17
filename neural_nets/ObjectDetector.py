import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch


class ObjectDetector(nn.Module):
    def __init__(self, baseModel, numClasses, deltacnn=False):
        super(ObjectDetector, self).__init__()
        self.feature_extractor = baseModel.features

        if deltacnn == True:
            baseModel.process_filters()
            self.feature_extractor = nn.Sequential(baseModel.sparsify, baseModel.features, baseModel.adaptive_avg_pooling, baseModel.densify)

        self.feature_extractor.out_channels = baseModel.last_channel

        # self.feature_extractor.process_filters()

        self.anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
        
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)
        
        self.faster_rcnn = FasterRCNN(self.feature_extractor, numClasses, rpn_anchor_generator=self.anchor_generator, box_roi_pool=self.roi_pooler)

    def forward(self, x, targets=None):

    #    x = x.to('cpu').contiguous(memory_format=torch.channels_last)
    #    return self.feature_extractor(x)
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