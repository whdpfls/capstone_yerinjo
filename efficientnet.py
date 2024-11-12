import torch
import torch.nn as nn
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNetModel, self).__init__()
        # EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights='DEFAULT')
        # 마지막 분류 레이어를 출력 클래스 수에 맞춤
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
