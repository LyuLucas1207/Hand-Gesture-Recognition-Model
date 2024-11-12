import sys, os
import utils.printtools as pt
from enums.model_dict import SVM_kernel, SVM_gamma
from sklearn.svm import SVC


class SupportVectorMachineBuilder:
    def __init__(self):
        # 默认配置
        self.__SVM_config = {
            "kernel": "rbf",  # 默认核函数
            "C": 1.0,  # 默认正则化强度
            "gamma": "scale",  # 默认gamma值
            "degree": 3,  # 默认多项式核的多项式度数
            "max_iter": -1,  # 默认最大迭代次数
            "random_state": None,  # 默认随机种子
            "model": None,
        }

    def set_kernel(self):
        pt.print_boxed_message("Set Kernel Function (kernel)")
        pt.print_boxed_options(SVM_kernel, "None/N. Default")
        while True:
            user_input = input("Select kernel (1-5, or 'none' for default): ").strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for kernel: 'rbf'")
                break
            elif user_input.isdigit() and int(user_input) in SVM_kernel:
                self.__SVM_config["kernel"] = SVM_kernel[int(user_input)]
                print(f"kernel set to: {SVM_kernel[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-5, or 'none'.")
        return self

    def set_C(self):
        pt.print_boxed_message("Set Regularization Parameter (C)")
        while True:
            user_input = input(
                "Enter the regularization parameter C (positive float, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for C: 1.0")
                break
            try:
                c_value = float(user_input)
                if c_value > 0:
                    self.__SVM_config["C"] = c_value
                    print(f"C set to: {c_value}")
                    break
                else:
                    print("Invalid input. Value must be positive.")
            except ValueError:
                print("Invalid input. Please enter a positive float or 'none'.")
        return self

    def set_gamma(self):
        pt.print_boxed_message("Set Gamma Value (gamma)")
        pt.print_boxed_options(SVM_gamma, "None/N. Default")
        while True:
            user_input = input("Select gamma (1 or 2, or 'none' for default): ").strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for gamma: 'scale'")
                break
            elif user_input.isdigit() and int(user_input) in SVM_gamma:
                self.__SVM_config["gamma"] = SVM_gamma[int(user_input)]
                print(f"gamma set to: {SVM_gamma[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1, 2, or 'none'.")
        return self

    def set_degree(self):
        pt.print_boxed_message("Set Degree (degree)")
        while True:
            user_input = input(
                "Enter the degree (positive integer, or 'none' for default, only applies to poly kernel): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for degree: 3")
                break
            if user_input.isdigit() and int(user_input) > 0:
                self.__SVM_config["degree"] = int(user_input)
                print(f"degree set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_max_iter(self):
        pt.print_boxed_message("Set Maximum Number of Iterations (max_iter)")
        while True:
            user_input = input(
                "Enter the maximum number of iterations (positive integer, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for max_iter: -1")
                break
            if user_input.isdigit() and int(user_input) > 0:
                self.__SVM_config["max_iter"] = int(user_input)
                print(f"max_iter set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_random_state(self):
        pt.print_boxed_message("Set Random Seed (random_state)")
        while True:
            user_input = input(
                "Enter the random seed (positive integer, or 'none' for default): "
            ).strip()
            if user_input.lower() in ["none", "n"]:
                print("Using default value for random_state: None")
                break
            if user_input.isdigit():
                self.__SVM_config["random_state"] = int(user_input)
                print(f"random_state set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self

    def set_SVM_model(self):
        pt.print_boxed_message("Building Support Vector Machine Model")
        self.__SVM_config["model"] = SVC(
            kernel=self.__SVM_config["kernel"],
            C=self.__SVM_config["C"],
            gamma=self.__SVM_config["gamma"],
            degree=self.__SVM_config["degree"],
            max_iter=self.__SVM_config["max_iter"],
            random_state=self.__SVM_config["random_state"],
        )
        return self

    def build(self):
        # 返回配置好的SVM配置
        return self.__SVM_config


# 创建工厂函数
def create_SVM_trainer():
    loop = True
    while loop:
        pt.print_boxed_message("Support Vector Machine Trainer Configuration")
        builder = SupportVectorMachineBuilder()
        config = (
            builder.set_kernel()
            .set_C()
            .set_gamma()
            .set_degree()
            .set_max_iter()
            .set_random_state()
            .set_SVM_model()
            .build()
        )
        pt.print_boxed_message("Support Vector Machine Configuration Completed")
        pt.print_aligned_model_config(config)

        pt.print_boxed_yes_no("Configuration Complete. Proceed?", "Yes", "Reset")
        last_check = input("Select (1/2/Other): ").strip()
        if last_check == "1":  # 确认配置无误
            loop = False
        elif last_check == "2":  # 重置配置
            continue
        else:  # 用户退出
            sys.exit("Program exited.")

    return config
