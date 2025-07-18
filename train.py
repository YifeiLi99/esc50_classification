import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data.esc50_dataset import ESC50Dataset
from models.cnn_model import SimpleCNN
import torchaudio

# 配置参数
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集路径
TRAIN_LIST = "./data/train_files.txt"
VAL_LIST = "./data/val_files.txt"

# Dataset & DataLoader
train_dataset = ESC50Dataset(TRAIN_LIST)
val_dataset = ESC50Dataset(VAL_LIST)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 模型
total_classes = 50
model = SimpleCNN(n_classes=total_classes).to(DEVICE)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 日志文件
log_file = open("train_log.txt", "w")

# 训练函数
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_correct = 0, 0
    for features, labels in dataloader:
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc

# 验证函数
def validate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc

if __name__ == "__main__":

    # 检测后端
    print(torchaudio.list_audio_backends())
    print(torchaudio.get_audio_backend())
    #确保权重文件夹存在
    os.makedirs("./weights", exist_ok=True)

    log_file.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    print(f"训练集类别数: {len(train_dataset.labels)}, 类别: {train_dataset.labels}")
    print(f"模型输出类别数: {total_classes}")

    # 主训练循环
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        log_line = f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        print(log_line)
        log_file.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")

        # 每个 epoch 存一次模型
        torch.save(model.state_dict(), f"./weights/epoch_{epoch+1}.pth")

    log_file.close()
    print("训练完成，模型已保存")
