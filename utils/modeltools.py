# Hand Gesture Recognition Model - Print Utility Functions
#
# This module provides utility functions for saving and loading machine learning models
# using the Python `pickle` library. These utilities ensure that models can be persistently
# stored on disk and later retrieved for inference or further training.

import os, torch
import pickle
import utils.filetools as ft


def save_model(model, path):
    directory = os.path.dirname(path)
    ft.create_dir(directory)
    
    with open(path, "wb") as f:
        pickle.dump({"model": model}, f)
        print(f"Model saved to '{path}'.")

def load_model(path):
    with open(path, "rb") as f:
        model_dict = pickle.load(f)
        model = model_dict["model"]

    return model, model_dict

def pytorch_save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def pytorch_load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Model loaded from {load_path}")
    return model

def pytorch_save_full_model(model, save_path):
    torch.save(model, save_path)
    print(f"Full model saved to {save_path}")

def pytorch_load_full_model(load_path, device="cpu"):
    model = torch.load(load_path, map_location=device)
    print(f"Full model loaded from {load_path}")
    return model.to(device)  # Ensure the model is moved to the correct device

def save_metrics(metrics, metrics_path):
    """
    保存评估指标到一个txt文件。

    Parameters:
    metrics (dict): 包含评估指标的字典，键是指标名称，值是对应的数值或矩阵。
    metrics_path (str): 保存文件的路径，必须是.txt文件。

    Example:
    metrics = {
        'Accuracy': 0.95,
        'F1 Score': 0.90,
        'Recall': 0.88,
        'Precision': 0.92,
        'Confusion Matrix': [[50, 5], [3, 42]]
    }
    """
    # 确保父目录存在
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as file:
        for metric, value in metrics.items():
            file.write(f"{metric}: ")
            if isinstance(value, (list, tuple)):
                file.write("\n")
                for row in value:
                    file.write("  " + " ".join(map(str, row)) + "\n")
            else:
                file.write(f"{value}\n")
    print(f"Metrics saved to {metrics_path}")
