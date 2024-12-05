import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from efficientnetB4 import EfficientNetB4
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 현재 시간 생성 함수
def get_current_time():
    return datetime.now().strftime("%m-%d-%H-%M")

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
        label = [int(x) for x in self.img_labels.iloc[idx, 1].split(',')]  # Multi-label 처리
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor


# Transform 정의
transform = transforms.Compose([
    transforms.Resize((380, 380)),  # EfficientNet-B4에 적합한 해상도
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터 경로
base_dataset_path = 'dataset/unified_spect_folds'
save_path = 'plots'
os.makedirs(save_path, exist_ok=True)

# Multi-label Accuracy 계산 함수
def calculate_multi_label_accuracy(y_true, y_pred_probs, top_k=2):
    correct = 0
    for true, pred_probs in zip(y_true, y_pred_probs):
        top_k_preds = np.argsort(pred_probs)[-top_k:]  # 상위 top_k 클래스 선택
        if any(true[i] == 1 and i in top_k_preds for i in range(len(true))):
            correct += 1
    return correct / len(y_true) * 100

# Confusion Matrix 시각화 함수
def plot_confusion_matrix(y_true, y_pred_probs, labels, title, save_path):
    # 예측된 Top-1 클래스
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_true, axis=1)  # Multi-label에서 가장 높은 값의 인덱스로 변환

    # Confusion Matrix 생성
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# Train 함수
def train(epoch, model, optimizer, criterion, trainloader):
    model.train()
    train_loss = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # 모델 출력
        outputs = model(inputs)

        # 손실 계산 (Multi-label)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(trainloader)
    return avg_loss

# Validation/Test 함수
def validate_or_test(model, criterion, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 모델 출력
            outputs = model(inputs)

            # 손실 계산
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # 예측 확률 저장
            preds = torch.sigmoid(outputs)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    multi_label_acc = calculate_multi_label_accuracy(all_targets, all_preds, top_k=2)
    return avg_loss, multi_label_acc, all_targets, all_preds

# Train, Validation, Test 구성
fold_performance = []
train_accs_all_folds = []
valid_accs_all_folds = []

for valid_fold in range(5):  # fold 0~4 중 valid 선택
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

    # EfficientNet B4 모델 학습
    print(f"\nTraining EfficientNet-B4 for Valid Fold {valid_fold}")
    model = EfficientNetB4(num_classes=11).to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    criterion = nn.BCEWithLogitsLoss()  # Multi-label Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_valid_acc = 0
    train_accs = []
    valid_accs = []

    for epoch in range(100):  # 에포크 변경
        train_loss = train(epoch, model, optimizer, criterion, trainloader)
        valid_loss, valid_acc, _, _ = validate_or_test(model, criterion, validloader)
        scheduler.step()

        train_accs.append(train_loss)
        valid_accs.append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

    train_accs_all_folds.append(train_accs)
    valid_accs_all_folds.append(valid_accs)

    fold_performance.append({
        'fold': valid_fold,
        'best_valid_acc': best_valid_acc
    })

# Fold별 Train/Validation Plot
plt.figure(figsize=(20, 15))
current_time = get_current_time()

for fold_idx in range(5):  # 총 5개의 Fold
    # Train Accuracy Plot
    plt.subplot(5, 2, fold_idx * 2 + 1)
    plt.plot(range(1, len(train_accs_all_folds[fold_idx]) + 1), train_accs_all_folds[fold_idx], label='Train Accuracy', color='blue')
    plt.title(f"Fold {fold_idx} - Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    # Validation Accuracy Plot
    plt.subplot(5, 2, fold_idx * 2 + 2)
    plt.plot(range(1, len(valid_accs_all_folds[fold_idx]) + 1), valid_accs_all_folds[fold_idx], label='Validation Accuracy', color='red')
    plt.title(f"Fold {fold_idx} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plot_filename = f"{current_time}_fold_accuracy_plots.png"
plt.savefig(os.path.join(save_path, plot_filename))
plt.close()

print(f"Combined Fold Accuracy Plot saved: {os.path.join(save_path, plot_filename)}")

# Test 단계
test_loss, test_acc, test_targets, test_preds = validate_or_test(model, criterion, testloader)

# Confusion Matrix 저장
labels = ["car_horn", "car", "scream", "gun_shot", "siren", "vehicle_siren", "fire", "skidding", "train", "explosion", "breaking"]
conf_matrix_path = os.path.join(save_path, f"{current_time}_confusion_matrix.png")
plot_confusion_matrix(test_targets, test_preds, labels, "Confusion Matrix for Test Set", conf_matrix_path)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
print(f"Confusion Matrix saved at: {conf_matrix_path}")
