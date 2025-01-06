from enum import Enum


class DataType(Enum):
    LAYER_BASED = 1  # CNN
    INSTANCE_BASED = 2  # MLP, DT, RF


class ModelType(Enum):
    CNN = 1
    MLP = 2
    DECISION_TREE = 3
    RANDOM_FOREST = 4
