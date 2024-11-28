import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetB0, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.efficientnet.dropout = nn.Dropout(p=0.4)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)