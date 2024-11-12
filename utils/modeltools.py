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
    with open(path, 'wb') as f:
        pickle.dump({'model': model}, f)
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
    with open(path, 'rb') as f:
        model_dict = pickle.load(f)
        model = model_dict['model']

    return model, model_dict
