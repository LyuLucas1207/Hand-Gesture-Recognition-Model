import sys, os
import utils.printtools as pt
from sklearn.neural_network import MLPClassifier
from enums.model_dict import NN_activation, NN_solver, NN_learning_rate


class SklearnNeuralNetworkBuilder:
    def __init__(self):
        self.__NN_config = {
            "hidden_layer_sizes": (100,),  # 默认隐藏层结构
            "activation": "relu",  # 默认激活函数
            "solver": "adam",  # 默认优化算法
            "alpha": 0.0001,  # 默认L2正则化强度
            "learning_rate": "constant",  # 默认学习率策略
            "max_iter": 200,  # 默认最大迭代次数
            "random_state": None,  # 默认随机种子
            "model": None,
        }

    def set_hidden_layer_sizes(self):
        pt.print_boxed_message("Set Hidden Layer Sizes (hidden_layer_sizes)")
        while True:
            user_input = input(
                "Enter hidden layer sizes as a comma-separated list (e.g., 128,64), or 'none' for default: "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for hidden_layer_sizes: (100,)")
                break
            try:
                # 将用户输入的逗号分隔字符串转化为整数元组
                layer_sizes = tuple(map(int, user_input.split(",")))
                if all(size > 0 for size in layer_sizes):  # 确保每一层的大小为正整数
                    self.__NN_config["hidden_layer_sizes"] = layer_sizes
                    print(f"hidden_layer_sizes set to: {layer_sizes}")
                    break
                else:
                    print("Invalid input. All layer sizes must be positive integers.")
            except ValueError:
                print(
                    "Invalid input. Please enter a valid comma-separated list of integers or 'none'."
                )
        return self

    def set_activation(self):
        pt.print_boxed_message("Set Activation Function (activation)")
        pt.print_boxed_options(NN_activation, "Other. Exit")
        while True:
            user_input = input(
                "Select activation function (1-4, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for activation: 'relu'")
                break
            elif user_input.isdigit() and int(user_input) in NN_activation:
                self.__NN_config["activation"] = NN_activation[int(user_input)]
                print(f"activation set to: {NN_activation[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-4, or 'none'.")
        return self

    def set_solver(self):
        pt.print_boxed_message("Set Solver Type (solver)")
        pt.print_boxed_options(NN_solver, "Other. Exit")
        while True:
            user_input = input("Select solver (1-3, or 'none' for default): ").strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for solver: 'adam'")
                break
            elif user_input.isdigit() and int(user_input) in NN_solver:
                self.__NN_config["solver"] = NN_solver[int(user_input)]
                print(f"solver set to: {NN_solver[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-3, or 'none'.")
        return self

    def set_alpha(self):
        pt.print_boxed_message("Set Regularization Strength (alpha)")
        while True:
            user_input = input(
                "Enter regularization strength (positive float, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for alpha: 0.0001")
                break
            try:
                alpha_value = float(user_input)
                if alpha_value > 0:
                    self.__NN_config["alpha"] = alpha_value
                    print(f"alpha set to: {alpha_value}")
                    break
                else:
                    print("Invalid input. Value must be positive.")
            except ValueError:
                print("Invalid input. Please enter a positive float or 'none'.")
        return self

    def set_learning_rate(self):
        pt.print_boxed_message("Set Learning Rate Schedule (learning_rate)")
        pt.print_boxed_options(NN_learning_rate, "Other. Exit")
        while True:
            user_input = input(
                "Select learning rate schedule (1-3, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for learning_rate: 'constant'")
                break
            elif user_input.isdigit() and int(user_input) in NN_learning_rate:
                self.__NN_config["learning_rate"] = NN_learning_rate[int(user_input)]
                print(f"learning_rate set to: {NN_learning_rate[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-3, or 'none'.")
        return self

    def set_max_iter(self):
        pt.print_boxed_message("Set Maximum Number of Iterations (max_iter)")
        while True:
            user_input = input(
                "Enter the maximum number of iterations (positive integer, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for max_iter: 200")
                break
            if user_input.isdigit() and int(user_input) > 0:
                self.__NN_config["max_iter"] = int(user_input)
                print(f"max_iter set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_random_state(self):
        pt.print_boxed_message("Set Random Seed (random_state)")
        while True:
            user_input = input(
                "Enter random seed (positive integer, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for random_state: None")
                break
            if user_input.isdigit():
                self.__NN_config["random_state"] = int(user_input)
                print(f"random_state set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_sklearn_NN_model(self):
        # 创建MLPClassifier对象
        pt.print_boxed_message("Building Neural Network Model")
        self.__NN_config["model"] = MLPClassifier(
            hidden_layer_sizes=self.__NN_config["hidden_layer_sizes"],
            activation=self.__NN_config["activation"],
            solver=self.__NN_config["solver"],
            alpha=self.__NN_config["alpha"],
            learning_rate=self.__NN_config["learning_rate"],
            max_iter=self.__NN_config["max_iter"],
            random_state=self.__NN_config["random_state"],
        )
        return self

    def build(self):
        self.set_sklearn_NN_model()
        return self.__NN_config


# 创建工厂函数
def create_NN_trainer():
    loop = True
    while loop:
        pt.print_boxed_message("Neural Network Trainer Configuration")
        builder = SklearnNeuralNetworkBuilder()
        NN_config = (
            builder.set_hidden_layer_sizes()
            .set_activation()
            .set_solver()
            .set_alpha()
            .set_learning_rate()
            .set_max_iter()
            .build()
        )
        pt.print_boxed_message("Neural Network Configuration Completed")
        pt.print_aligned_model_config(NN_config)
        pt.print_boxed_yes_no("Configuration Complete. Proceed?", "Yes", "Reset")
        last_check = input("Select (1/2/Other): ").strip()
        if last_check == "1":  # 确认配置无误
            loop = False
        elif last_check == "2":  # 重置配置
            continue
        else:  # 用户退出
            sys.exit("Program exited.")

    return NN_config
