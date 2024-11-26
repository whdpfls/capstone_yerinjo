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
import random
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
        file_name = self.img_labels.iloc[idx, 0]

        # 파일 이름에 경로가 포함되어 있다면 img_dir를 추가하지 않음
        if os.path.isabs(file_name):
            img_path = file_name
        else:
            img_path = os.path.join(self.img_dir, file_name)

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

# 데이터 경로
base_dataset_path = 'dataset/unified_spect_folds'
save_path = 'plots'
os.makedirs(save_path, exist_ok=True)

# Training 함수
def train(epoch, model, optimizer, criterion, trainloader):
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
def test(epoch, model, criterion, testloader):
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
for fold_num in range(5):  # fold 0~4 중 train/valid 랜덤 선택
    # Fold 설정
    valid_fold = fold_num
    train_folds = [f for f in range(5) if f != valid_fold]
    random.shuffle(train_folds)
    train_fold = train_folds[0]

    train_annot_path = os.path.join(base_dataset_path, f"{train_fold}_train_annot.csv")
    valid_annot_path = os.path.join(base_dataset_path, f"{valid_fold}_valid_annot.csv")
    train_spect_path = os.path.join(base_dataset_path, f"fold_{train_fold}")
    valid_spect_path = os.path.join(base_dataset_path, f"fold_{valid_fold}")
    test_annot_path = os.path.join(base_dataset_path, "test_annot.csv")
    test_spect_path = os.path.join(base_dataset_path, "fold_5")

    # 데이터셋 및 데이터로더
    trainset = AudioSpectrogramDataset(train_annot_path, train_spect_path, transform=transform)
    validationset = AudioSpectrogramDataset(valid_annot_path, valid_spect_path, transform=transform)
    testset = AudioSpectrogramDataset(test_annot_path, test_spect_path, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    validloader = DataLoader(validationset, batch_size=32, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # 모델별 결과 저장
    train_acc_results = {}
    test_acc_results = {}

    for version, color in zip(['b0', 'b4', 'b7'], ['blue', 'red', 'yellow']):
        print(f"\nTraining EfficientNet-{version.upper()} for Train Fold {train_fold} and Valid Fold {valid_fold}")
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
            train_acc, train_loss = train(epoch, model, optimizer, criterion, trainloader)
            valid_acc, valid_loss = test(epoch, model, criterion, validloader)

            train_acc_list.append(train_acc)
            test_acc_list.append(valid_acc)

            print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Train Loss: {train_loss:.4f}, Valid Acc: {valid_acc:.2f}%, Valid Loss: {valid_loss:.4f}")

        train_acc_results[version] = train_acc_list
        test_acc_results[version] = test_acc_list

    # Plot 저장
    epochs = range(1, 81)

    # Train Accuracy Plot
    plt.figure(figsize=(14, 7))
    for version, color in zip(['b0', 'b4', 'b7'], ['blue', 'red', 'yellow']):
        plt.plot(epochs, train_acc_results[version], label=f'Train Acc {version.upper()}', color=color)
    plt.title(f'Train Accuracy: Train Fold {train_fold}, Valid Fold {valid_fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'train_accuracy_fold_{train_fold}_valid_{valid_fold}.png'))
    plt.close()

    # Valid Accuracy Plot
    plt.figure(figsize=(14, 7))
    for version, color in zip(['b0', 'b4', 'b7'], ['blue', 'red', 'yellow']):
        plt.plot(epochs, test_acc_results[version], label=f'Valid Acc {version.upper()}', color=color)
    plt.title(f'Valid Accuracy: Train Fold {train_fold}, Valid Fold {valid_fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'valid_accuracy_fold_{train_fold}_valid_{valid_fold}.png'))
    plt.close()
