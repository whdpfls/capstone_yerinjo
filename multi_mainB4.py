import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from efficientnetB4 import EfficientNetB4
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

dataset_path = 'dataset/unified_spect_folds'
save_path = 'plots'
os.makedirs(save_path, exist_ok=True)

# 현재 시간 생성 함수
def get_current_time():
    from datetime import datetime
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

        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])  # Single-label 처리
        if self.transform:
            image = self.transform(image)

        return image, label

# Transform 정의
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Confusion Matrix 기반 오분류 데이터 추출
def get_misclassified_data(y_true, y_pred, dataset):
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    misclassified_data = [dataset[i] for i in misclassified_indices]
    return misclassified_data

# Confusion Matrix 시각화
def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# Accuracy Plot
def plot_accuracy(train_acc, valid_acc, fold, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy", color="blue")
    plt.plot(range(1, len(valid_acc) + 1), valid_acc, label="Validation Accuracy", color="red")
    plt.title(f"Accuracy for Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Validation/Test 함수
def validate_or_test(model, criterion, dataloader, dataset, misclassified_collector=None):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Confusion Matrix 및 오분류 데이터
    cm = confusion_matrix(y_true, y_pred)
    if misclassified_collector is not None:
        misclassified_collector.extend(get_misclassified_data(y_true, y_pred, dataset))

    return total_loss / len(dataloader), cm, y_true, y_pred

# 모델 학습 및 반복
def train_and_test_model(folds=5, target_accuracy=90):
    for fold in range(folds):
        current_time = get_current_time()  # 현재 시간 생성
        train_acc, valid_acc = [], []

        train_annotation_file = os.path.join(dataset_path, f"{fold}_train_annot.csv")
        valid_annot_path = os.path.join(dataset_path, f"{fold}_valid_annot.csv")
        test_annot_path = os.path.join(dataset_path, "test_annot.csv")

        # 데이터셋 생성
        trainset = AudioSpectrogramDataset(train_annotation_file, dataset_path, transform=transform)
        validationset = AudioSpectrogramDataset(valid_annot_path, dataset_path, transform=transform)
        testset = AudioSpectrogramDataset(test_annot_path, dataset_path, transform=transform)

        train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(validationset, batch_size=32, shuffle=False)
        test_loader = DataLoader(testset, batch_size=32, shuffle=False)

        model = EfficientNetB4(num_classes=11).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_valid_accuracy = 0

        # 학습
        for epoch in range(100):
            model.train()
            correct_train = 0
            total_train = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                # Train Accuracy 계산
                preds = outputs.argmax(dim=1)
                correct_train += preds.eq(targets).sum().item()
                total_train += targets.size(0)

            train_acc_epoch = correct_train / total_train * 100
            train_acc.append(train_acc_epoch)

            # Validation Accuracy
            valid_loss, valid_cm, _, _ = validate_or_test(model, criterion, valid_loader, validationset)
            valid_accuracy = np.diag(valid_cm).sum() / valid_cm.sum() * 100
            valid_acc.append(valid_accuracy)

            print(f"Fold {fold}, Epoch {epoch}, Train Accuracy: {train_acc_epoch:.2f}%, Valid Accuracy: {valid_accuracy:.2f}%")

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy

            # 목표 정확도 달성 시 종료
            if valid_accuracy >= target_accuracy:
                break

        # Accuracy Plot 저장
        accuracy_plot_path = os.path.join(save_path, f"{current_time}_fold_{fold}_accuracy.png")
        plot_accuracy(train_acc, valid_acc, fold, accuracy_plot_path)
        print(f"Accuracy Plot saved for Fold {fold}: {accuracy_plot_path}")

        # Test 단계
        misclassified_data = []
        test_loss, test_cm, y_true, y_pred = validate_or_test(model, criterion, test_loader, testset, misclassified_data)
        test_accuracy = np.diag(test_cm).sum() / test_cm.sum() * 100

        print(f"Test Accuracy for Fold {fold}: {test_accuracy:.2f}%")
        confusion_matrix_path = os.path.join(save_path, f"{current_time}_fold_{fold}_confusion_matrix.png")
        plot_confusion_matrix(test_cm, ["car_horn", "car", "scream", "gun_shot", "siren",
                                        "vehicle_siren", "fire", "skidding", "train",
                                        "explosion", "breaking"],
                              f"Confusion Matrix for Fold {fold}", confusion_matrix_path)
        print(f"Confusion Matrix saved for Fold {fold}: {confusion_matrix_path}")

        # 오분류 데이터 재검증
        if misclassified_data:
            re_test_path = os.path.join(save_path, f"{current_time}_fold_{fold}_re_test_confusion_matrix.png")
            _, re_test_cm, _, _ = validate_or_test(model, criterion, DataLoader(misclassified_data, batch_size=32, shuffle=False), misclassified_data)
            plot_confusion_matrix(
                re_test_cm,
                ["car_horn", "car", "scream", "gun_shot", "siren",
                 "vehicle_siren", "fire", "skidding", "train",
                 "explosion", "breaking"],
                f"Re-Test Confusion Matrix for Fold {fold}",
                re_test_path
            )
            print(f"Re-Test Confusion Matrix saved for Fold {fold}: {re_test_path}")

train_and_test_model()
