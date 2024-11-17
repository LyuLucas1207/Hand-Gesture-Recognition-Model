import sys, os

from enums.model_dict import trainer_models, trainer_scalers, trainer_metrics
import utils.filetools as ft
import utils.printtools as pt
from base.SupportVectorMachineBuilder import create_SVM_trainer
from base.RandomForestBuilder import create_RF_trainer
from base.KNearestNeighborsBuilder import create_KNN_trainer
from base.LogisticRegressionBuilder import create_LR_trainer
from base.SklearnNeuralNetworkBuilder import create_NN_trainer


class ModelTrainerBuilder:
    def __init__(self):
        self.__model_config = {
            "trainer_model": None,
            "specific_model": None,
            "test_size": 0.2,
            "random_seed": None,
            "if_scaler": False,
            "trainer_scaler": None,
            "if_pipeline": False,
            "if_save_model": False,
            "model_name": None,
            "trainer_metric": None,
            "if_save_metrics": False,
            "metrics_name": None,
        }

    def set_trainer_model(self):
        tm = {
            1: "Random Forest",
            2: "K-Nearest Neighbors",
            3: "Logistic Regression",
            4: "Support Vector Machine",
            5: "Sklearn Neural Network",
        }
        pt.print_boxed_message("Select Model")
        pt.print_boxed_options(tm, "Other. Exit")

        model = input(
            "Select model: "
        ).strip()  # Get user input and remove leading/trailing whitespace
        self.set_select(model, trainer_models, "trainer_model")
        return self

    def set_specific_model(self):
        pt.print_boxed_message("Set Specific Model")
        trainer_model = self.__model_config["trainer_model"]
        trainer_config_creators = {
            trainer_models[1]: create_RF_trainer,
            trainer_models[2]: create_KNN_trainer,
            trainer_models[3]: create_LR_trainer,
            trainer_models[4]: create_SVM_trainer,
            trainer_models[5]: create_NN_trainer,
        }
        if trainer_model in trainer_config_creators:
            self.__model_config["specific_model"] = trainer_config_creators[
                trainer_model
            ]()
        else:
            sys.exit("Program exited.")

        return self

    def set_test_size(self):
        pt.print_boxed_message("Set Test Size")
        while True:
            test_size = input("Enter test size (0.0 - 1.0): ").strip()
            if not test_size.replace(".", "", 1).isdigit():
                sys.exit("Invalid input. Program exited.")
            test_size = float(test_size)
            if 0.0 < test_size < 1.0:
                self.__model_config["test_size"] = test_size
                print(f"Test size set to {test_size}.")
                break
            else:
                print(
                    "Invalid range. Please enter a value between 0.0 and 1.0 (exclusive)."
                )
        return self

    def set_random_seed(self):
        pt.print_boxed_message("Set Random Seed")
        while True:
            random_seed = input("Enter random seed: ").strip()
            if not random_seed.isdigit():
                sys.exit("Invalid input. Program exited.")
            random_seed = int(random_seed)
            self.__model_config["random_seed"] = random_seed
            print(f"Random seed set to {random_seed}.")
            break
        return self

    def set_if_scaler(self):
        pt.print_boxed_yes_no("Set if Scaler")
        if_scaler = input("Select if scaler (1/2): ").strip()
        self.set_if(if_scaler, "if_scaler")
        return self

    def set_trainer_scaler(self):
        if self.__model_config["if_scaler"]:
            ts = {
                1: "Standard Scaler",
                2: "Min-Max Scaler",
                3: "Robust Scaler",
                4: "Max Abs Scaler",
                5: "Normalizer",
            }
            pt.print_boxed_message("Select Scaler")
            pt.print_boxed_options(ts, "Other. Exit")

            scalers = input("Select scaler: ").strip()
            self.set_select(scalers, trainer_scalers, "trainer_scaler")
        return self

    def set_if_pipeline(self):
        pt.print_boxed_yes_no("Set if Pipeline")
        if_pipeline = input("Select if pipeline (1/2): ").strip()
        self.set_if(if_pipeline, "if_pipeline")
        return self

    def set_if_save_model(self):
        pt.print_boxed_yes_no("Set if Save Model")
        if_save_model = input("Select if save model (1/2): ").strip()
        self.set_if(if_save_model, "if_save_model")
        return self

    def set_model_name(self):
        if self.__model_config["if_save_model"]:
            pt.print_boxed_message("Set Model Name")
            loop = True
            while loop:
                model_name = input(
                    "Enter a custom model name (without extension): "
                ).strip()
                # Sanitize user input
                if not model_name:
                    print("Model name cannot be empty. Please try again.")
                    continue
                model_path = os.path.join(f"{model_name}.p")
                if os.path.exists(model_path):
                    print(
                        f"The file '{model_path}' already exists. Please choose a different name."
                    )
                else:
                    self.__model_config["model_name"] = model_path
                    print(f"Model name set to: {model_path}")
                loop = self.set_name(model_path, "model_name")

        return self

    def set_trainer_metric(self):
        tm = {
            1: "Accuracy Score",
            2: "F1 Score",
            3: "Confusion Matrix",
            4: "Recall Score",
            5: "Precision Score",
            6: "ALL",
        }
        pt.print_boxed_message("Select Metric")
        pt.print_boxed_options(tm, "Other. Exit")

        metrics = input("Select metric: ").strip()
        self.set_select(metrics, trainer_metrics, "trainer_metric")
        return self

    def set_if_save_metrics(self):
        pt.print_boxed_yes_no("Set if Save Metrics")
        if_save_metrics = input("Select if save metrics (1/2): ").strip()
        self.set_if(if_save_metrics, "if_save_metrics")
        return self

    def set_metrics_name(self):
        if self.__model_config["if_save_metrics"]:
            pt.print_boxed_message("Set Metrics Name")
            loop = True
            while loop:
                metrics_name = input(
                    "Enter a custom metrics name (without extension): "
                ).strip()
                if not metrics_name:
                    print("Metrics name cannot be empty. Please try again.")
                    continue
                metrics_path = os.path.join(f"{metrics_name}.txt")
                if os.path.exists(metrics_path):
                    print(
                        f"The file '{metrics_path}' already exists. Please choose a different name."
                    )
                else:
                    self.__model_config["metrics_name"] = metrics_path
                    print(f"Metrics name set to: {metrics_path}")
                    loop = False
        return self

    def build(self):
        # return ModelTrainerBuilder(**self.__model_config)
        return self.__model_config

    def set_if(self, flag, flag_name):
        if flag == "1":
            self.__model_config[flag_name] = True
        elif flag == "2":
            self.__model_config[flag_name] = False
        else:
            sys.exit("Program exited")

    def set_select(self, select, dictionary, key):
        if select.isdigit() and int(select) in dictionary.keys():
            self.__model_config[key] = dictionary[int(select)]
            print(f"Set to: {dictionary[int(select)]}")
        else:
            sys.exit("Invalid input. Program exited.")
        return self

    def set_name(self, path, name):
        if os.path.exists(path):
            print(f"File '{path}' already exists. Please choose a different name.")
        else:
            self.__model_config[name] = path
            print(f"Name set to: {path}")

        return False


def create_model_trainer():
    loop = True
    while loop:
        pt.print_boxed_message("Model Trainer Configuration")
        MTB = ModelTrainerBuilder()
        model_config = (
            MTB.set_trainer_model()
            .set_specific_model()
            .set_test_size()
            .set_random_seed()
            .set_if_scaler()
            .set_trainer_scaler()
            .set_if_pipeline()
            .set_if_save_model()
            .set_model_name()
            .set_trainer_metric()
            .set_if_save_metrics()
            .set_metrics_name()
            .build()
        )

        pt.print_boxed_message("Model Configuration")
        pt.print_aligned_model_config(model_config)
        pt.print_boxed_yes_no("Last Check", "Continue", "Reset")
        last_check = input("Select (1/2/Other): ").strip()
        if last_check == "1":
            loop = False
        elif last_check == "2":
            continue
        else:
            sys.exit("Program exited.")

    return model_config
