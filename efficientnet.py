import torch
import torch.nn as nn
from torchvision import models


class EfficientNetModel(nn.Module):
    def __init__(self, model_version='b0', num_classes=10):
        super(EfficientNetModel, self).__init__()

        # 모델 버전에 따라 EfficientNet 모델 선택
        if model_version == 'b0':
            self.efficientnet = models.efficientnet_b0(weights='DEFAULT')
        elif model_version == 'b4':
            self.efficientnet = models.efficientnet_b4(weights='DEFAULT')
        elif model_version == 'b7':
            self.efficientnet = models.efficientnet_b7(weights='DEFAULT')
        else:
            raise ValueError(f"Unsupported model version: {model_version}")

        # 마지막 분류 레이어를 출력 클래스 수에 맞춤
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.efficientnet(x)
