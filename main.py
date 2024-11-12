import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from efficientnet import EfficientNetModel  # EfficientNet 모델 가져오기
import time
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Accuracy를 저장할 딕셔너리
train_accuracies_dict = {'b0': [], 'b4': [], 'b7': []}
test_accuracies_dict = {'b0': [], 'b4': [], 'b7': []}

# 사용자 정의 데이터셋 클래스
class AudioSpectrogramDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotation_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((216, 216)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 경로 및 파일
train_annot_path = '/dataset/us8k_train.csv'
validation_annot_path = '/dataset/us8k_valid.csv'
train_spect_path = '/dataset/us8k_train'
validation_spect_path = '/dataset/us8k_valid'

# 데이터셋 및 데이터로더
trainset = AudioSpectrogramDataset(train_annot_path, train_spect_path, transform=transform)
validationset = AudioSpectrogramDataset(validation_annot_path, validation_spect_path, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testloader = DataLoader(validationset, batch_size=32, shuffle=False, num_workers=4)

# Training 함수
def train(epoch, model, optimizer, criterion):
    model.train()
    correct = 0
    total = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc

# 검증 함수
def test(epoch, model, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc

# 모델 학습 및 결과 저장
for version in ['b0', 'b4', 'b7']:
    print(f"\nTraining EfficientNet-{version.upper()}")
    model = EfficientNetModel(model_version=version, num_classes=10)
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    train_acc_list = []
    test_acc_list = []

    for epoch in range(80):  # 에포크 수를 조정
        train_acc = train(epoch, model, optimizer, criterion)
        test_acc = test(epoch, model, criterion)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    train_accuracies_dict[version] = train_acc_list
    test_accuracies_dict[version] = test_acc_list

# Plotting
epochs = range(1, 11)
plt.figure(figsize=(14, 7))

for version, color in zip(['b0', 'b4', 'b7'], ['blue', 'red', 'yellow']):
    plt.plot(epochs, train_accuracies_dict[version], label=f'Train Acc {version.upper()}', color=color, linestyle='--')
    plt.plot(epochs, test_accuracies_dict[version], label=f'Test Acc {version.upper()}', color=color)

plt.title('Training and Test Accuracy for EfficientNet B0, B4, B7')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
