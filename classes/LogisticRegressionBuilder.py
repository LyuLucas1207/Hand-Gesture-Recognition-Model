import sys, os
import utils.printtools as pt
from enums.model_dict import LR_penalty, LR_solver
from sklearn.linear_model import LogisticRegression

class LogisticRegressionBuilder:
    def __init__(self):
        self.__LR_config = {
            'penalty': 'l2',                # 默认正则化类型
            'solver': 'lbfgs',              # 默认求解器
            'C': 1.0,                       # 默认正则化强度
            'max_iter': 100,                # 默认最大迭代次数
            'multi_class': 'multinomial',   # 默认使用Softmax多分类
            'random_state': None,           # 默认随机种子
            'model': None
        }

    def set_penalty(self):
        pt.print_boxed_message("Set Penalty Type (penalty)")
        pt.print_boxed_options(LR_penalty, "None/N. Default")
        while True:
            user_input = input("Select penalty type (1-4, or 'none' for default): ").strip()
            if user_input.lower() in ['none', 'n']:
                print("Using default value for default: 'l2'")
                break
            elif user_input.isdigit() and int(user_input) in LR_penalty:
                self.__LR_config['penalty'] = LR_penalty[int(user_input)]
                print(f"penalty set to: {LR_penalty[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-4, or 'none'.")
        return self

    def set_solver(self):
        pt.print_boxed_message("Set Solver Type (solver)")
        pt.print_boxed_options(LR_solver, "None/N. Default")
        while True:
            user_input = input("Select solver (1-5, or 'none' for default): ").strip()
            if user_input.lower() in ['none', 'n']:
                print("Using default value for default: 'lbfgs'")
                break
            elif user_input.isdigit() and int(user_input) in LR_solver:
                self.__LR_config['solver'] = LR_solver[int(user_input)]
                print(f"solver set to: {LR_solver[int(user_input)]}")
                break
            else:
                print("Invalid input. Please enter 1-5, or 'none'.")
        return self

    def set_regularization_strength(self):
        pt.print_boxed_message("Set Regularization Strength (C)")
        while True:
            user_input = input("Enter the regularization strength (positive float, or 'none' for default): ").strip()
            if user_input.lower() in ['none', 'n']:
                print("Using default value for regularization strength: 1.0")
                break
            try:
                c_value = float(user_input)
                if c_value > 0:
                    self.__LR_config['C'] = c_value
                    print(f"Regularization strength set to: {c_value}")
                    break
                else:
                    print("Invalid input. Value must be positive.")
            except ValueError:
                print("Invalid input. Please enter a positive float or 'none'.")
        return self

    def set_max_iter(self):
        pt.print_boxed_message("Set Maximum Number of Iterations (max_iter)")
        while True:
            user_input = input("Enter the maximum number of iterations (positive integer, or 'none' for default): ").strip()
            if user_input.lower() in ['none', 'n']:
                print("Using default value for max_iter: 100")
                break
            if user_input.isdigit() and int(user_input) > 0:
                self.__LR_config['max_iter'] = int(user_input)
                print(f"max_iter set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self
    
    def set_random_state(self):
        pt.print_boxed_message("Set Random State (random_state)")
        while True:
            user_input = input("Enter the random state (positive integer, or 'none' for default): ").strip()
            if user_input.lower() in ['none', 'n']:
                print("Using default value for random state: None")
                break
            if user_input.isdigit():
                self.__LR_config['random_state'] = int(user_input)
                print(f"random state set to: {user_input}")
                break
            else:
                print("Invalid input. Please enter a positive integer or 'none'.")
        return self
    
    def set_LR_model(self):
        pt.print_boxed_message("Building Logistic Regression Model")
        self.__LR_config['model'] = LogisticRegression(penalty=self.__LR_config['penalty'],
                                                       solver=self.__LR_config['solver'],
                                                       C=self.__LR_config['C'],
                                                       max_iter=self.__LR_config['max_iter'],
                                                       random_state=self.__LR_config['random_state'])
        return self


    def build(self):
        return self.__LR_config
    


# 创建工厂函数
def create_LR_trainer():
    loop = True
    while loop:
        pt.print_boxed_message("Logistic Regression Trainer Configuration")
        builder = LogisticRegressionBuilder()
        config = builder.set_penalty()\
                        .set_solver()\
                        .set_regularization_strength()\
                        .set_max_iter()\
                        .set_random_state()\
                        .set_LR_model()\
                        .build()
        pt.print_boxed_message("Logistic Regression Configuration Completed")
        pt.print_aligned_model_config(config)
        pt.print_boxed_yes_no("Configuration Complete. Proceed?", "Yes", "Reset")
        last_check = input("Select (1/2/Other): ").strip()
        if last_check == '1':  # 确认配置无误
            loop = False
        elif last_check == '2':  # 重置配置
            continue
        else:  # 用户退出
            sys.exit("Program exited.")

    return config
