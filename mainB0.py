import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from efficientnetB0 import EfficientNetB0
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 현재 시간 생성 함수
def get_current_time():
    return datetime.now().strftime("%m-%d-%H-%M")

# Accuracy를 저장할 딕셔너리
train_accuracies_dict = []
valid_accuracies_dict = []
test_accuracy = 0

# 최고 성능 정보를 저장할 딕셔너리
best_performance = {}

# 사용자 정의 데이터셋 클래스
class AudioSpectrogramDataset(Dataset):
    def __init__(self, annotation_file, base_path, transform=None):
        self.img_labels = pd.read_csv(annotation_file, names=['file_name', 'label'])
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        file_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.base_path, file_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


# Transform 정의
transform = transforms.Compose([
    transforms.Resize((128, 87)),
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
def test(model, criterion, testloader):
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


# Train, Valid, Test 구성 및 학습
for valid_fold in range(5):  # fold 0~4 중 valid 선택
    train_folds = [f for f in range(5) if f != valid_fold]
    train_annotation_file = os.path.join(base_dataset_path, f"{valid_fold}_train_annot.csv")
    valid_annot_path = os.path.join(base_dataset_path, f"{valid_fold}_valid_annot.csv")
    test_annot_path = os.path.join(base_dataset_path, "test_annot.csv")

    # 데이터셋 생성
    trainset = AudioSpectrogramDataset(train_annotation_file, base_dataset_path, transform=transform)
    validationset = AudioSpectrogramDataset(valid_annot_path, base_dataset_path, transform=transform)
    testset = AudioSpectrogramDataset(test_annot_path, base_dataset_path, transform=transform)

    # DataLoader
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    validloader = DataLoader(validationset, batch_size=32, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    # EfficientNet B0 모델 학습
    print(f"\nTraining EfficientNet-B0 for Valid Fold {valid_fold}")
    model = EfficientNetB0(num_classes=11).to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_train_acc = 0
    best_valid_acc = 0
    best_epoch = 0

    train_acc_list = []
    valid_acc_list = []

    for epoch in range(50):
        train_acc, train_loss = train(epoch, model, optimizer, criterion, trainloader)
        valid_acc, valid_loss = test(model, criterion, validloader)
        scheduler.step()

        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Valid Acc: {valid_acc:.2f}%")

        if valid_acc > best_valid_acc:
            best_train_acc = train_acc
            best_valid_acc = valid_acc
            best_epoch = epoch + 1

    test_acc, test_loss = test(model, criterion, testloader)
    test_accuracy = test_acc

    best_performance = {
        'best_train_acc': best_train_acc,
        'best_valid_acc': best_valid_acc,
        'best_epoch': best_epoch,
        'test_acc': test_accuracy
    }

    train_accuracies_dict.append(train_acc_list)
    valid_accuracies_dict.append(valid_acc_list)

# Plot 저장
epochs = range(1, 51)

# Train & Valid Accuracy Plot
plt.figure(figsize=(14, 7))
plt.plot(epochs, train_accuracies_dict[0], label=f'Train Acc B0', color='blue', linestyle='-')
plt.plot(epochs, valid_accuracies_dict[0], label=f'Valid Acc B0', color='blue', linestyle='--')
plt.title('Train and Valid Accuracy for B0')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
current_time = get_current_time()
plt.savefig(os.path.join(save_path, f'{current_time}_b0_train_valid_accuracy.png'))
plt.close()

# Test Accuracy Bar Plot
plt.figure(figsize=(14, 7))
plt.bar(['B4'], [test_accuracy], color=['blue'])
plt.title('Test Accuracy for B0')
plt.xlabel('Model')
plt.ylabel('Accuracy (%)')
current_time = get_current_time()
plt.savefig(os.path.join(save_path, f'{current_time}_b0_test_accuracy_bar_plot.png'))
plt.close()

# Best performance 출력
print("\nBest Performance Summary:")
print(f"EfficientNet-B0: Best Train Acc: {best_performance['best_train_acc']:.2f}%, Best Valid Acc: {best_performance['best_valid_acc']:.2f}%, Test Acc: {best_performance['test_acc']:.2f}%, Best Epoch: {best_performance['best_epoch']}")
