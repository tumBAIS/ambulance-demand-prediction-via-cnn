from typing import Any, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import ELU, LeakyReLU, ReLU
from tensorflow.keras.losses import MSE, Loss
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from tensorflow.keras.optimizers import SGD, Adam


def get_activation(name: str) -> Optional[Union[LeakyReLU, ReLU, ELU]]:
    """Returns keras activation object based on provided name (String)

    Parameters:
    name: str
        name of activation function (supported functions are 'leakyrelu', 'relu', 'elu', 'linear')

    Returns:
    activation: Optional[Union[LeakyReLU, ReLU, ELU]]
        corresponding Activation
    """

    if name == "leakyrelu":
        return LeakyReLU()
    if name == "relu":
        return ReLU()
    if name == "elu":
        return ELU()
    if name == "linear":
        return None
    raise ValueError(name, ": no valid activation name")


def get_optimizer(name: str, learning_rate: float) -> Union[SGD, Adam]:
    """Returns keras optimizer object based on provided name (String)

    Parameters:
    name: str
        name of optimizer (supported functions are 'SGD', 'Adam')

    Returns:
    Optimizer: Union[SGD, Adam]
        corresponding Optimizer
    """

    if name == "SGD":
        return SGD(learning_rate=learning_rate)
    if name == "Adam":
        return Adam(learning_rate=learning_rate)
    raise ValueError(name, ": no valid loss function name")


def get_loss_function(name: str) -> MSE:
    """Returns keras loss function based on provided name (String)

    Parameters:
    name: str
        name of loss function (metric) (supported functions is 'MSE')

    Returns:
    loss function: MSE
        corresponding tensorflow.keras.losses object
    """

    if name == "MSE":
        return MSE

    if name == "Custom_MSE":
        return CustomMSE()

    raise ValueError(name, ": no valid loss function name")


class CustomMSE(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true: np.ndarray, y_pred: np.ndarray):

        return tf.reduce_mean(
            tf.math.multiply(
                tf.math.square(y_pred - y_true), tf.math.square(y_true + 1)
            ),
            axis=-1,
        )


def get_metrics(*names) -> List[Union[MeanSquaredError, RootMeanSquaredError]]:
    """Returns metrics on provided names (Strings)

    Parameters:
    names: Tuple[str]
        name of metrics (supported metrics is 'MSE', 'RMSE')

    Returns:
    list of metrics: List[Union[MeanSquaredError, RootMeanSquaredError]]
        corresponding list of tensorflow.keras.losses / tf.keras.metrics objects
    """

    metrics = []
    for name in names:
        if name == "MSE":
            metrics.append(MeanSquaredError())
        elif name == "RMSE":
            metrics.append(RootMeanSquaredError())
        else:
            raise ValueError(name, ": no valid metric name")
    return metrics


def flatten_nested_list(nested_list: List[Any]) -> List[Any]:
    """Flattens nestes list ([[1,2,3,4],[[5,6],[7,8]]] -> [1,2,3,4,5,6,7,8])

    Paramters:
    nested_list: List[Any]
        Nested list to be flattened

    Returns:
    flattened_list: List[Any]
        Flattened list
    """
    flattened_list = []
    for l in nested_list:
        if isinstance(l, list):
            flattened_list.extend(flatten_nested_list(l))
        else:
            flattened_list.append(l)
    return flattened_list
