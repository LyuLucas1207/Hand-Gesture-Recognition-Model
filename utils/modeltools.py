# Hand Gesture Recognition Model - Print Utility Functions
#
# This module provides utility functions for saving and loading machine learning models
# using the Python `pickle` library. These utilities ensure that models can be persistently
# stored on disk and later retrieved for inference or further training.
#
# Functions:
# !1. save_model(model, path):
#     - Saves a machine learning model to the specified file path.
#     - If the directory does not exist, it will be created automatically using the `utils.filetools.create_dir` utility.
#
# !2. load_model(path):
#     - Loads a machine learning model from the specified file path.
#     - Returns the loaded model object and the entire dictionary containing the model.
#
# # Save the model
# mt.save_model(model, './models/random_forest_model.p')
#
# # Load the model
# loaded_model, model_dict = mt.load_model('./models/random_forest_model.p')
# print("Model loaded successfully:", loaded_model)


import os
import pickle
import utils.filetools as ft


def save_model(model, path):
    """
    Save a machine learning model to a specified path using pickle.
    If the folder or file does not exist, it will be created.

    Parameters:
    model (object): The model to be saved.
    path (str): The file path where the model will be saved.

    Returns:
    None
    """
    # Get the directory from the given path
    directory = os.path.dirname(path)

    # Check if the directory exists, create it if it doesn't
    ft.create_dir(directory)

    # Save the model to the specified path
    with open(path, "wb") as f:
        pickle.dump({"model": model}, f)
        print(f"Model saved to '{path}'.")


def load_model(path):
    """
    Load a machine learning model from a specified path using pickle.

    Parameters:
    path (str): The file path where the model is saved.

    Returns:
    object: The loaded model.
    dict: The dictionary containing the loaded model.
    """
    with open(path, "rb") as f:
        model_dict = pickle.load(f)
        model = model_dict["model"]

    return model, model_dict


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
