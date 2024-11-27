import sys, os
import utils.printtools as pt
from enums.model_dict import KNN_weights, KNN_algorithm
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighborsBuilder:
    def __init__(self):
        self.__KNN_config = {
            "n_neighbors": 5,  # 默认值
            "weights": "uniform",  # 默认值
            "algorithm": "auto",  # 默认值
            "leaf_size": 30,  # 默认值
            "model": None,
        }

    def set_n_neighbors(self):
        pt.print_boxed_message("Set Number of Neighbors (n_neighbors)")
        while True:
            user_input = input(
                "Enter the number of neighbors (positive integer, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for n_neighbors: 5")
                break
            elif user_input.isdigit() and int(user_input) > 0:
                self.__KNN_config["n_neighbors"] = int(user_input)
                print(f"n_neighbors set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_weights(self):
        pt.print_boxed_message("Set Weight Function (weights)")
        pt.print_boxed_options(options=KNN_weights, additional_option="None/N. Default")
        while True:
            user_input = input(
                "Select weights (1 for 'uniform', 2 for 'distance', or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for weights: 'uniform'")
                break
            elif user_input.isdigit() and int(user_input) in KNN_weights:
                self.__KNN_config["weights"] = KNN_weights[int(user_input)]
                print(f"weights set to: {KNN_weights[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1, 2, or 'none'.")
        return self

    def set_algorithm(self):
        pt.print_boxed_message("Set Algorithm (algorithm)")
        pt.print_boxed_options(
            options=KNN_algorithm, additional_option="None/N. Default"
        )
        while True:
            user_input = input(
                "Select algorithm (1-4, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for algorithm: 'auto'")
                break
            elif user_input.isdigit() and int(user_input) in KNN_algorithm:
                self.__KNN_config["algorithm"] = KNN_algorithm[int(user_input)]
                print(f"algorithm set to: {KNN_algorithm[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-4, or 'none'.")
        return self

    def set_leaf_size(self):
        pt.print_boxed_message("Set Leaf Size (leaf_size)")
        while True:
            user_input = input(
                "Enter the leaf size (positive integer, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for default: 30")
                break
            elif user_input.isdigit() and int(user_input) > 0:
                self.__KNN_config["leaf_size"] = int(user_input)
                print(f"leaf_size set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_KNN_model(self):
        pt.print_boxed_message("Building K-Nearest Neighbors Model")
        self.__KNN_config["model"] = KNeighborsClassifier(
            n_neighbors=self.__KNN_config["n_neighbors"],
            weights=self.__KNN_config["weights"],
            algorithm=self.__KNN_config["algorithm"],
            leaf_size=self.__KNN_config["leaf_size"],
        )
        return self

    def build(self):
        # 返回配置好的KNN配置
        return self.__KNN_config


# 创建一个工厂函数，便于使用
def create_KNN_trainer():
    loop = True
    while loop:
        pt.print_boxed_message("K-Nearest Neighbors Trainer Configuration")
        builder = KNearestNeighborsBuilder()
        KNN_config = (
            builder.set_n_neighbors()
            .set_weights()
            .set_algorithm()
            .set_leaf_size()
            .set_KNN_model()
            .build()
        )
        pt.print_boxed_message("K-Nearest Neighbors Configuration Completed")
        pt.print_aligned_model_config(KNN_config)
        pt.print_boxed_yes_no("Configuration Complete. Proceed?", "Yes", "Reset")
        last_check = input("Select (1/2/Other): ").strip()
        if last_check == "1":  # 确认配置无误
            loop = False
        elif last_check == "2":  # 重置配置
            continue
        else:  # 用户退出
            sys.exit("Program exited.")

    return KNN_config
