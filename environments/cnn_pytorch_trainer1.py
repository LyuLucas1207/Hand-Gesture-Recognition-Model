# How to run this code: python -m environments.cnn_pytorch_trainer1
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 加载手部关键点数据
with open("./data/data.pickle", "rb") as f:
    data_dict = pickle.load(f)

print(data_dict.keys())  # dict_keys(['data', 'labels'])

print(data_dict["data"][:1])  # 打印前 5 项以查看具体内容
print([type(sample) for sample in data_dict["data"]])  # 查看每个样本的类型

lengths = [len(sample) for sample in data_dict["data"]]
print("Sample lengths:", lengths[:10])  # 查看前 10 个样本的长度
print("Unique lengths:", set(lengths))  # 检查所有样本长度是否一致


def truncate_samples(data, target_length=42):
    """
    将所有超过 target_length 的样本截取到 target_length
    """
    truncated_data = []
    for sample in data:
        if len(sample) > target_length:
            truncated_data.append(sample[:target_length])  # 截取前 42 个特征
        else:
            truncated_data.append(sample)  # 保留原始样本
    return truncated_data

# 截取数据
data = truncate_samples(data_dict["data"], target_length=42)
data = np.asarray(data)

# data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

print(f"Data shape: {data.shape}")

# 调整数据形状以匹配 CNN
num_samples, feature_size = data.shape
data = data.reshape(num_samples, 1, 6, 7)  # 假设 feature_size = 6x7
print(f"Data shape after reshaping: {data.shape}")

# One-hot 编码（等价于 TensorFlow 的 `to_categorical`）
num_classes = len(np.unique(labels))
labels = np.eye(num_classes)[labels.astype(int)]  # 使用 numpy 实现 one-hot 编码

# 数据集分割
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=np.argmax(labels, axis=1)
)

# 转换为 PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 构造 DataLoader
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型结构
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),
        )

        # 计算 Flatten 大小
        test_input = torch.zeros((1, 1, 6, 7))  # 输入的样例形状
        flatten_size = self._get_flatten_size(test_input)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def _get_flatten_size(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=num_classes).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses = []
val_losses = []

## 模型训练
epochs = 30
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets.argmax(dim=1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    # 验证模型
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.argmax(dim=1))

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_accuracy += (preds == targets.argmax(dim=1)).float().mean().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.argmax(dim=1).cpu().numpy())

    val_losses.append(val_loss / len(test_loader))

    # 计算分类报告
    report = classification_report(all_targets, all_preds, target_names=[f"Class {i}" for i in range(num_classes)], zero_division=0)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy/len(test_loader):.4f}")
    print(report)

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")

plt.xticks(range(1, epochs + 1, 5))  # 每隔 5 个 epoch 设置一个刻度
plt.yticks(np.arange(2.6, max(max(train_losses), max(val_losses)) + 0.05, 0.05))  # y 轴从 2 开始，每隔 0.1

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# 测试模型
model.eval()
test_accuracy = 0.0
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)

        test_accuracy += (preds == targets.argmax(dim=1)).float().mean().item()

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.argmax(dim=1).cpu().numpy())

test_accuracy /= len(test_loader)
print(f"Final Model Accuracy: {test_accuracy * 100:.2f}%")

# 打印最终分类报告
final_report = classification_report(all_targets, all_preds, target_names=[f"Class {i}" for i in range(num_classes)], zero_division=0)
print("Final Classification Report:")
print(final_report)

# 保存模型
torch.save(model.state_dict(), "./models/cnn_advanced.pth")
print("Model saved as ./models/cnn_advanced.pth")

# 保存完整模型（包括结构和权重）
torch.save(model, "./models/cnn_advanced_complete.pth")
print("Model saved as ./models/cnn_advanced_complete.pth")