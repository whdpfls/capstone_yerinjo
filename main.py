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

# 최고 성능 정보를 저장할 딕셔너리
best_performance = {}

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
train_annot_path = 'dataset/ru_spect/train_annot.csv'
validation_annot_path = 'dataset/ru_spect/valid_annot.csv'
train_spect_path = 'dataset/ru_spect/trainset'
validation_spect_path = 'dataset/ru_spect/validset'

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
    train_loss = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = train_loss / len(trainloader)
    return acc, avg_loss

# 검증 함수
def test(epoch, model, criterion):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    return acc, avg_loss

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

    best_test_acc = 0
    best_epoch = 0
    best_train_loss = 0
    best_test_loss = 0

    train_acc_list = []
    test_acc_list = []

    for epoch in range(80):  # 에포크 수를 조정
        train_acc, train_loss = train(epoch, model, optimizer, criterion)
        test_acc, test_loss = test(epoch, model, criterion)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}')

        # 최고 정확도 갱신 시 저장
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_test_loss = test_loss
            best_lr = optimizer.param_groups[0]['lr']
            best_performance[version] = {
                'epoch': best_epoch,
                'learning_rate': best_lr,
                'train_loss': best_train_loss,
                'test_loss': best_test_loss,
                'test_accuracy': best_test_acc
            }

    train_accuracies_dict[version] = train_acc_list
    test_accuracies_dict[version] = test_acc_list

# 최고 성능 출력 (한 번만 출력)
print("\nBest Performance Summary:")
for version, performance in best_performance.items():
    print(f"\nEfficientNet-{version.upper()}:")
    print(f"Epoch: {performance['epoch']}")
    print(f"Learning Rate: {performance['learning_rate']}")
    print(f"Train Loss: {performance['train_loss']:.4f}")
    print(f"Test Loss: {performance['test_loss']:.4f}")
    print(f"Test Accuracy: {performance['test_accuracy']:.2f}%")

# Plotting
save_path = '/data/dpfls5645/repos/capstone_yerinjo/plots'
os.makedirs(save_path, exist_ok=True)

epochs = range(1, 81)
plt.figure(figsize=(14, 7))

for version, color in zip(['b0', 'b4', 'b7'], ['blue', 'red', 'yellow']):
    plt.plot(epochs, train_accuracies_dict[version], label=f'Train Acc {version.upper()}', color=color, linestyle='--')
    plt.plot(epochs, test_accuracies_dict[version], label=f'Test Acc {version.upper()}', color=color)

plt.title('Training and Test Accuracy for EfficientNet B0, B4, B7')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# 플롯 저장 함수
def save_plot_as_png(plot, filename):
    filepath = os.path.join(save_path, filename)
    plot.savefig(filepath, format='png')
    print(f"Plot saved at {filepath}")

# 플롯을 .png로 저장
save_plot_as_png(plt, 'training_plot.png')
plt.show()
