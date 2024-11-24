# How to run: python -m environments.simple_trainer        
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
    precision_score,
)

import utils.modeltools as mt
from base.ModelTrainerBuilder import create_model_trainer
from enums.model_dict import trainer_models


def trainer():
    model_config = create_model_trainer()
    data_dict = pickle.load(open("./data/data.pickle", "rb"))
    print(data_dict.keys())  # dict_keys(['data', 'labels'])

    data = np.asarray(data_dict["data"])
    labels = np.asarray(data_dict["labels"])

    test_size = model_config["test_size"] if "test_size" in model_config else 0.2

    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, shuffle=True, stratify=labels
    )
    trainer_model = model_select(model_config)
    if_scaler = model_config["if_scaler"]
    if_pipeline = model_config["if_pipeline"]
    if_save_model = model_config["if_save_model"]
    model_name = model_config["model_name"]
    if_save_metrics = model_config["if_save_metrics"]
    metrics_name = model_config["metrics_name"]

    if if_scaler:
        trainer_scaler = model_config["trainer_scaler"]()
        if if_pipeline:
            model = make_pipeline(trainer_scaler, trainer_model)
        else:
            model = Pipeline([("scaler", trainer_scaler()), ("model", trainer_model())])
    else:
        model = trainer_model

    print(f"Training {model}...")
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    if model_config["trainer_metric"] == "ALL":
        metrics = {
            "Accuracy": accuracy_score(y_test, y_predict),
            "F1 Score": f1_score(y_test, y_predict, average="weighted"),
            "Recall": recall_score(y_test, y_predict, average="weighted"),
            "Precision": precision_score(y_test, y_predict, average="weighted"),
            "Confusion Matrix": confusion_matrix(y_test, y_predict).tolist(),
        }
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    else:
        metric_function = model_config["trainer_metric"]
        metrics = {metric_function.__name__: metric_function(y_test, y_predict)}

    if if_save_metrics:
        mt.save_metrics(metrics, f"./output/{metrics_name}")
    if if_save_model:
        mt.save_model(model, f"./models/{model_name}")


def model_select(model_config):
    trainer_model = model_config["trainer_model"]
    print(f"Selected model: {trainer_model}")
    print(f"Available models: {trainer_models[1]}")
    print(f"Available models: {trainer_models[2]}")
    print(f"Available models: {trainer_models[3]}")
    print(f"Available models: {trainer_models[4]}")
    print(f"Available models: {trainer_models[5]}")
    if trainer_model == trainer_models[1]:
        return model_config["specific_model"]["model"]
    elif trainer_model == trainer_models[2]:
        return model_config["specific_model"]["model"]
    elif trainer_model == trainer_models[3]:
        return model_config["specific_model"]["model"]
    elif trainer_model == trainer_models[4]:
        return model_config["specific_model"]["model"]
    elif trainer_model == trainer_models[5]:
        return model_config["specific_model"]["model"]
    else:
        sys.exit("Program exited due to invalid model selection.")


if __name__ == "__main__":
    trainer()
