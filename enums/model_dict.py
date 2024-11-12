from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
    precision_score,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
)


trainer_models = {
    1: RandomForestClassifier,
    2: KNeighborsClassifier,
    3: LogisticRegression,
    4: SVC,
    5: MLPClassifier,
}

trainer_kernel_functions = {
    1: "linear",
    2: "poly",
    3: "rbf",
    4: "sigmoid",
}

trainer_scalers = {
    1: StandardScaler,
    2: MinMaxScaler,
    3: RobustScaler,
    4: MaxAbsScaler,
    5: Normalizer,
}

trainer_metrics = {
    1: accuracy_score,
    2: f1_score,
    3: confusion_matrix,
    4: recall_score,
    5: precision_score,
    6: "ALL",
}

RF_criterion = {
    1: "gini",
    2: "entropy",
    3: "log_loss",
}

KNN_weights = {
    1: "uniform",
    2: "distance",
}

KNN_algorithm = {
    1: "auto",
    2: "ball_tree",
    3: "kd_tree",
    4: "brute",
}

LR_penalty = {
    1: "l1",
    2: "l2",
    3: "elasticnet",
}

LR_solver = {
    1: "newton-cg",
    2: "lbfgs",
    3: "liblinear",
    4: "sag",
    5: "saga",
}

SVM_kernel = {
    1: "linear",
    2: "poly",
    3: "rbf",
    4: "sigmoid",
    5: "precomputed",
}

SVM_gamma = {
    1: "scale",
    2: "auto",
}

NN_activation = {
    1: "identity",
    2: "logistic",
    3: "tanh",
    4: "relu",
}

NN_solver = {
    1: "lbfgs",
    2: "sgd",
    3: "adam",
}

NN_learning_rate = {
    1: "constant",
    2: "invscaling",
    3: "adaptive",
}
