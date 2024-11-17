import sys, os
import utils.printtools as pt
from enums.model_dict import RF_criterion
from sklearn.ensemble import RandomForestClassifier


class RandomForestBuilder:
    def __init__(self):
        # 随机森林配置字典
        self.__RF_config = {
            "n_estimators": 100,  # 树的数量
            "max_depth": None,  # 树的最大深度
            "criterion": "gini",  # 分裂标准
            "random_state": None,  # 随机种子
            "model": None,  # 模型
        }

    def set_n_estimators(self):
        """
        设置树的数量 (n_estimators)
        """
        pt.print_boxed_message("Set Number of Trees (n_estimators)")
        while True:
            n_estimators = input(
                "Enter the number of trees (positive integer or 'None'): "
            ).strip()
            if n_estimators.isdigit() and int(n_estimators) > 0:
                self.__RF_config["n_estimators"] = int(n_estimators)
                print(f"Number of trees set to: {n_estimators}.")
                break
            elif n_estimators.lower() in ["none", "n"]:
                print("Number of trees set to default: 100.")
                break
            else:
                print("Invalid input. Please enter a positive integer.")

        return self

    def set_max_depth(self):
        """
        设置树的最大深度 (max_depth)
        """
        pt.print_boxed_message("Set Maximum Depth of Trees (max_depth)")
        while True:
            max_depth = input(
                "Enter the maximum depth of the trees (positive integer or 'None'): "
            ).strip()
            if max_depth.isdigit() and int(max_depth) > 0:
                self.__RF_config["max_depth"] = int(max_depth)
                print(f"Maximum depth set to: {max_depth}.")
                break
            elif max_depth.lower() in ["none", "n"]:
                print("Maximum depth set to default: None.")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'None'.")

        return self

    def set_criterion(self):
        """
        设置分裂标准 (criterion)
        """
        pt.print_boxed_message("Set Splitting Criterion")
        pt.print_boxed_options(
            options=RF_criterion, additional_option="None/N. Default"
        )

        while True:
            criterion = input("Select splitting criterion(1 is default): ").strip()
            if criterion.lower() in ["none", "n"]:
                print("Criterion set to default: 'gini'.")
                break
            elif criterion.isdigit() and int(criterion) in RF_criterion:
                self.__RF_config["criterion"] = RF_criterion[int(criterion)]
                print(f"Splitting criterion set to: {RF_criterion[int(criterion)]}.")
                break
            else:
                print("Invalid input. Please select.")

        return self

    def set_random_state(self):
        """
        设置随机种子 (random_state)
        """
        pt.print_boxed_message("Set Random State (For Random Forest)")
        while True:
            random_state = input("Enter random state (integer or 'None'): ").strip()
            if random_state.isdigit():
                self.__RF_config["random_state"] = int(random_state)
                print(f"Random state set to: {random_state}.")
                break
            elif random_state.lower() in ["none", "n"]:
                print("Random state set to default: None.")
                break
            else:
                print("Invalid input. Please enter an integer or 'None'.")

        return self

    def set_RF_model(self):
        pt.print_boxed_message("Building Random Forest Model")
        self.__RF_config["model"] = RandomForestClassifier(
            n_estimators=self.__RF_config["n_estimators"],
            max_depth=self.__RF_config["max_depth"],
            criterion=self.__RF_config["criterion"],
            random_state=self.__RF_config["random_state"],
        )
        return self

    def build(self):
        """
        返回配置好的随机森林参数字典
        """
        return self.__RF_config


def create_RF_trainer():
    """
    创建一个随机森林训练器配置
    """
    loop = True
    while loop:
        pt.print_boxed_message("Random Forest Configuration")
        RFB = RandomForestBuilder()
        RF_config = (
            RFB.set_n_estimators()
            .set_max_depth()
            .set_criterion()
            .set_random_state()
            .set_RF_model()
            .build()
        )

        pt.print_boxed_message("Random Forest Configuration Summary")
        pt.print_aligned_model_config(RF_config)

        pt.print_boxed_yes_no("Configuration Complete. Proceed?", "Yes", "Reset")
        last_check = input("Select (1/2/Other): ").strip()
        if last_check == "1":  # 确认配置无误
            loop = False
        elif last_check == "2":  # 重置配置
            continue
        else:  # 用户退出
            sys.exit("Program exited.")

    return RF_config
