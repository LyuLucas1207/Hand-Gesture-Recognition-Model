# How to run: python -m environments.cnn_pytorch_trainer2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils.modeltools import pytorch_save_full_model
from core.CNNPyTorchBuilder import CNNModel, train_model

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001

    # Data preprocessing and loading
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.Resize((128, 128)),                # 保证图片尺寸一致
        transforms.ToTensor(),                        # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5])   # 灰度图归一化，范围 [-1, 1]
    ])

    dataset = datasets.ImageFolder("./data/images", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 训练集加载器. shuffle=True 表示打乱数据
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss function, and optimizer
    model = CNNModel(num_classes = 36, # 26个字母 + 10个数字
                     in_channels = 1, 
                     out_channels = 32, 
                     image_width = 128, 
                     image_height = 128
                     )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs = 10,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Save the trained model
    pytorch_save_full_model(trained_model, "./models/cnn_model_pytorch.pth")

if __name__ == "__main__":
    main()
