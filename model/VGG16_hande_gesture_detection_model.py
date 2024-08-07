import torch
import torch.nn as nn
import torchvision.models as models

class HandGestureVGG16(nn.Module):
    def __init__(self, num_classes=29):
        super(HandGestureVGG16, self).__init__()
        # Pre-trained VGG16 model
        self.vgg16 = models.vgg16(weights='IMAGENET1K_V1')

        # change last layer for our task
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)