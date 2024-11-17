"""
    Formula to calculate the output size of a convolutional layer:
    Output = (Input - Kernel + 2 * Padding) / Stride + 1

    Formula to calculate the output size of a pooling layer:
    Output = (Input - Kernel) / Stride + 1

    For example:
    - Input size: 128x128 (HxW) in_channels=3
    - Convolutional layer with kernel size 3, stride 1, and padding 1

    Output size:
    - H_out = (128 - 3 + 2 * 1) / 1 + 1 = 128
    - W_out = (128 - 3 + 2 * 1) / 1 + 1 = 128
    - Output size: 128x128
"""

"""
    Pooling layer:
        - Max Pooling: MaxPool2d(kernel_size, stride)
            - This pooling will get the maximum value in the kernel_size x kernel_size window.
            -e.g., MaxPool2d(2, 2) (128x128 -> 64x64)
        - Average Pooling: AvgPool2d(kernel_size, stride)
            - This pooling will get the average value in the kernel_size x kernel_size window.
            -e.g., AvgPool2d(2, 2) (128x128 -> 64x64)
        - Global Pooling: AdaptiveAvgPool2d(output_size)
            - This pooling will reduce the size of the input to the output_size.
            -e.g., AdaptiveAvgPool2d(64) (128x128 -> 64x64)

    Activation layer:
        - ReLU: ReLU()
            - This activation function will set all negative values to zero.
            -e.g., ReLU() (128x128 -> 128x128) (x = max(0, x), x < 0 -> 0, x >= 0 -> x)
        - Sigmoid: Sigmoid()
            - This activation function will squash the output between 0 and 1.
            -e.g., Sigmoid() (128x128 -> 128x128) (x = 1 / (1 + exp(-x)), value between 0 and 1)
        - Leaky ReLU: LeakyReLU(negative_slope)
            - This activation function will set all negative values to negative_slope * x.
            -e.g., LeakyReLU(0.01) (128x128 -> 128x128) (x = max(0.01 * x, x), x < 0 -> 0.01 * x, x >= 0 -> x)
        - Tanh: Tanh()
            - This activation function will squash the output between -1 and 1.
            -e.g., Tanh() (128x128 -> 128x128) (x = (exp(x) - exp(-x)) / (exp(x) + exp(-x)), value between -1 and 1)
        - Softmax: Softmax(dim)
            - This activation function will squash the output between 0 and 1 and normalize the values to sum to 1.
            -e.g., Softmax(dim=1) (128x128 -> 128x128) (x = exp(x) / sum(exp(x)), sum of all values = 1)

    Fully connected layer:
        - Linear: Linear(in_features, out_features)
            - This layer will perform a linear transformation of the input.
            -e.g., Linear(128, 64) (128x128 -> 64x64)
        - Flatten: Flatten()
            - This layer will flatten the input to a 1D tensor.
            -e.g., Flatten() (128x128 -> 16384)

    Dropout layer:
        - Dropout: Dropout(p)
            - This layer will randomly set a fraction p of the input units to zero.
            -e.g., Dropout(0.5) (128x128 -> 128x128)

    Normalization layer:
        - Batch Normalization: BatchNorm2d(num_features)
            - This layer will normalize the input across the batch, to keep the mean-value close to 0 and the variance close to 1
            - Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
            -e.g., BatchNorm2d(64) (128x128 -> 128x128) 
        - Layer Normalization: LayerNorm(normalized_shape)
            - This layer will normalize the input across the features.
            - Formula: y = (x - mean) / var * gamma + beta
            -e.g., LayerNorm(128) (128x128 -> 128x128)
        - Instance Normalization: InstanceNorm2d(num_features)
            - This layer will normalize the input across the features.
            -e.g., InstanceNorm2d(64) (128x128 -> 128x128)

    Convolutional layer:
        - Conv2d: Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            - This layer will perform a 2D convolution on the input.
            -e.g., Conv2d(3, 64, kernel_size=3, stride=1, padding=1) (128x128 -> 128x128)

"""

import os
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes, in_channels=3, out_channels = 32, image_width=128, image_height=128):
        super(CNNModel, self).__init__()
        self.nc = num_classes
        self.ic = in_channels
        self.oc = out_channels
        self.iw = image_width
        self.ih = image_height

        # 第一层卷积 + 激活 + 池化
        # iw x ih x 3 -> iw x ih x 32
        self.conv1 = nn.Conv2d(in_channels = self.ic, out_channels = self.oc, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.Tanh()
        # iw x ih x 32 -> iw/2 x ih/2 x 32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # 第二层卷积 + 激活 + 池化
        # iw/2 x ih/2 x oc -> iw/2 x ih/2 x (2 * oc)
        self.conv2 = nn.Conv2d(in_channels = self.oc, out_channels = 2 * self.oc, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.Tanh()
        # iw/2 x ih/2 x 64 -> iw/4 x ih/4 x 64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # 全连接层部分
        self.flatten = nn.Flatten() # 展平, if 32x32x64 -> 65536
        self.fc1 = nn.Linear(2 * self.oc * (self.iw // 4) * (self.ih // 4), 256)  # 输入维度, 输出维度
        self.relu_fc1 = nn.Tanh()  # 激活
        self.dropout = nn.Dropout(0.5)  # Dropout
        self.fc2 = nn.Linear(256, num_classes)  # 输出类别数
        
    def forward(self, x):
        # 第一层卷积块
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 第二层卷积块
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 全连接层
        x = self.flatten(x)  # 展平
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Print training statistics
        train_acc = 100.0 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

        # Validate the model
        validate_model(model, val_loader, criterion, device)

    print("Training completed.")
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient tracking
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Statistics
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100.0 * correct / total
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_acc:.2f}%")
    return val_acc

