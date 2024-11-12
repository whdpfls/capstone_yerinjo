import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import os
import argparse
import time
import matplotlib.pyplot as plt
from efficientnet import EfficientNetModel  # EfficientNet 모델 가져오기

# Argument Parser 설정
parser = argparse.ArgumentParser(description='PyTorch Spectrogram training with EfficientNet')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # learning rate를 0.01로 낮추어 시작합니다
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args(args=[])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

best_acc = 0
start_epoch = 0

# Accuracy와 Loss를 저장할 리스트
train_accuracies = []
train_losses = []
test_accuracies = []
test_losses = []

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
    transforms.Resize((216, 216)),  # EfficientNet b0 입력 크기
    transforms.RandomHorizontalFlip(),  # 데이터 증강
    transforms.RandomRotation(10),  # 데이터 증강
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # EfficientNet에서 사용하는 정규화 값
])

# 데이터 경로 및 파일
train_annot_path = '/Users/yerin/Desktop/24_2/capstonecode/dataset/us8k_train.csv'
validation_annot_path = '/Users/yerin/Desktop/24_2/capstonecode/dataset/us8k_valid.csv'
train_spect_path = '/Users/yerin/Desktop/24_2/capstonecode/dataset/us8k_train'
validation_spect_path = '/Users/yerin/Desktop/24_2/capstonecode/dataset/us8k_valid'

# 데이터셋 및 데이터로더
trainset = AudioSpectrogramDataset(train_annot_path, train_spect_path, transform=transform)
validationset = AudioSpectrogramDataset(validation_annot_path, validation_spect_path, transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testloader = DataLoader(validationset, batch_size=32, shuffle=False, num_workers=4)

# EfficientNet 모델 초기화
net = EfficientNetModel(num_classes=10)  # EfficientNet 모델로 변경
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# 체크포인트
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 손실 함수 및 최적화 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)  # AdamW로 변경
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

## Training 함수
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        start_time = time.time()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 10 == 0:
            batch_processing_time = time.time() - start_time
            print(f'Batch {batch_idx + 1}/{len(trainloader)} processed in {batch_processing_time:.3f} sec')
        elif (batch_idx == len(trainloader) - 1):
            print(f'Train Epoch {epoch}, Batch {batch_idx + 1}, Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}% ({correct}/{total})')

    # 에포크 종료 후 평균 손실과 정확도 저장
    train_losses.append(train_loss / len(trainloader))
    train_accuracies.append(100. * correct / total)

# 검증 함수
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx == len(testloader) - 1:
                print(f'Test Epoch {epoch}, Loss: {test_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

    # 에포크 종료 후 평균 손실과 정확도 저장
    test_losses.append(test_loss / len(testloader))
    test_accuracies.append(100. * correct / total)

    # 체크포인트 저장
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

# 학습 및 검증 실행
if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 80):
        train(epoch)
        test(epoch)
        scheduler.step()


