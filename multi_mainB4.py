import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnetB4 import EfficientNetB4
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.metrics import confusion_matrix
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 현재 시간 생성 함수
def get_current_time():
    return datetime.now().strftime("%m-%d-%H-%M")

# Accuracy 저장용 리스트
fold_performance = []
test_accuracy = 0

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
    transforms.Resize((380, 380)),  # EfficientNet-B4에 적합한 해상도 - 380
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


# 다중 레이블 평가를 포함한 테스트 함수
def test_multilabel(model, criterion, testloader):
    model.eval()
    total = 0
    test_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 다중 레이블 예측
            prob = torch.sigmoid(outputs)
            topk_values, topk_indices = prob.topk(2, dim=1)
            correct_preds = torch.any(topk_indices == targets.unsqueeze(1), dim=1)

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            total += targets.size(0)
            correct += correct_preds.sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(topk_indices.cpu().numpy())

    acc = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    return acc, avg_loss, all_preds, all_targets


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
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    validloader = DataLoader(validationset, batch_size=64, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # EfficientNet B4 모델 학습
    print(f"\nTraining EfficientNet-B4 for Valid Fold {valid_fold}")
    model = EfficientNetB4(num_classes=11).to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.BCEWithLogitsLoss()  # 다중 레이블 예측을 위한 손실 함수
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_accs = []
    valid_accs = []

    for epoch in range(100):  # 에포크 변경
        train_acc, train_loss = train(epoch, model, optimizer, criterion, trainloader)
        valid_acc, valid_loss, _, _ = test_multilabel(model, criterion, validloader)
        scheduler.step()

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        # 10번째 에포크마다 출력
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, Valid Acc: {valid_acc:.2f}%")

    # Plot Fold Accuracy
    current_time = get_current_time()
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(valid_accs, label='Valid Accuracy')
    plt.title(f"Accuracy for Fold {valid_fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plot_filename = f"{current_time}-b4-{valid_fold}.png"
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()

# Test 단계
test_acc, test_loss, test_preds, test_targets = test_multilabel(model, criterion, testloader)

# Confusion Matrix Plot
conf_matrix = confusion_matrix(test_targets, np.concatenate(test_preds))
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(11), yticklabels=range(11))
plt.title("Confusion Matrix for Test Set")
plt.xlabel("Predicted")
plt.ylabel("True")
conf_matrix_filename = f"{get_current_time()}-confusion-matrix.png"
plt.savefig(os.path.join(save_path, conf_matrix_filename))
plt.close()

print(f"Test Accuracy: {test_acc:.2f}%")
