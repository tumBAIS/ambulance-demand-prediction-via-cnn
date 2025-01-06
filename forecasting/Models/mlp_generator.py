import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import yaml
from Data_Handler.data_storage import DataStorage, generate_training_test_data
from Models.helper import (get_activation, get_loss_function, get_metrics,
                           get_optimizer)
from Models.types import DataType, ModelType
from Results_Handler.results_handler import BestModel, output_results
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

FIXED_SETTINGS_FILE = "./Settings/MLP/settings_mlp_fixed.yaml"

# pylint: disable=no-member, too-many-instance-attributes
class MLP:
    def __init__(self, **kwargs):

        # Set statically defined hyperparameters via yaml file
        # replace such that files can be named by id
        self.id = str(datetime.today()).replace(":", "-")
        with open(FIXED_SETTINGS_FILE, "r") as stream:
            try:
                fixed_settings = yaml.safe_load(stream)
                for key, value in fixed_settings.items():
                    setattr(self, key, value)
            except yaml.YAMLError as exc:
                print(exc)

        # Dynamically defined hyperparameters via Settings/Bayesian Optimization (settings_mlp.csv)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # overwrite string values read from settings file by corresponding objects
        self.metrics = get_metrics(*self.metrics)
        self.optimizer = get_optimizer(self.optimizer, self.learning_rate)
        self.loss_function = get_loss_function(self.loss_function_name)
        self.activation_1 = get_activation(self.activation_1)
        self.activation_2 = get_activation(self.activation_2)
        self.activation_3 = get_activation(self.activation_3)
        self.activation_4 = get_activation(self.activation_4)
        self.activation_outputs = get_activation(self.activation_outputs)

        # dictionary with all mlp attributes for printing results
        dic_attributes = self.__dict__.copy()
        del dic_attributes["model"]
        self.data = list(dic_attributes.values())
        self.data_headers = list(dic_attributes.keys())

    def build_mlp_model(self, X: np.ndarray) -> models.Model:

        sample_shape = len(X[0])

        inputs = layers.Input(shape=sample_shape)

        # hidden layer 1
        d1 = layers.Dense(units=self.nrUnits_1)(inputs)
        if self.activation_1 is not None:
            d1 = self.activation_1(d1)
        dropout = layers.Dropout(self.dropout_1)(d1)

        # hidden layer 2
        if self.nrLayers >= 2:
            d2 = layers.Dense(units=self.nrUnits_2)(dropout)
            if self.activation_2 is not None:
                d2 = self.activation_2(d2)
            dropout = layers.Dropout(self.dropout_2)(d2)

        # hidden layer 3
        if self.nrLayers >= 3:
            d3 = layers.Dense(units=self.nrUnits_3)(dropout)
            if self.activation_3 is not None:
                d3 = self.activation_3(d3)
            dropout = layers.Dropout(self.dropout_3)(d3)

        # hidden layer 4
        if self.nrLayers >= 4:
            d4 = layers.Dense(units=self.nrUnits_4)(dropout)
            if self.activation_4 is not None:
                d4 = self.activation_4(d4)
            dropout = layers.Dropout(self.dropout_4)(d4)

        # output layer
        outputs = layers.Dense(units=1)(dropout)
        if self.activation_outputs is not None:
            outputs = self.activation_outputs(outputs)

        model = models.Model(inputs=[inputs], outputs=outputs)

        self.model = model

        return model


# pylint: disable=too-many-arguments
def run_MLP(
    settings: Dict[str, Any],
    data_storage: DataStorage,
    incident_types: List[str],
    best_model: BestModel = None,
    X: np.ndarray = None,
    y: np.ndarray = None,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    output_path: str = "./Results",
):

    # generate MLP
    mlp = MLP(**settings)
    modeltype = ModelType.MLP
    data_type = DataType.INSTANCE_BASED
    features_dict = data_storage.get_features_dict(data_type, True)  # already filtered

    # generate training and testset
    if X_train is None or y_train is None or X_test is None or y_test is None:

        # get data (if necessary, i.e. X any y not given)
        # check if data (X and y) was saved in data_storage, if yes, get this data
        if (
            (X is None or y is None)
            and (data_type in data_storage.X)
            and (data_type in data_storage.y)
        ):
            X = data_storage.X[data_type]  # already filtered
            y = data_storage.y[data_type]  # already filtered

        else:
            raise ValueError("Please provide X or y or store data in data_storage.")

        X_train, y_train, X_test, y_test = generate_training_test_data(
            *X,
            y=y,
            test_size=mlp.test_size,
            random_state=mlp.random_state,
            data_type=data_type,
            shuffle_data=mlp.shuffle_data,
        )

    model = mlp.build_mlp_model(X)

    if mlp.early_stopping:
        callbacks = [
            EarlyStopping(
                monitor=mlp.early_stopping_monitor,
                patience=mlp.early_stopping_patience,
                restore_best_weights=mlp.restore_best_weights,
            )
        ]

    model.compile(loss=mlp.loss_function, optimizer=mlp.optimizer, metrics=mlp.metrics)

    # fit model and track time
    start = time.time()
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=mlp.batch_size,
        epochs=mlp.no_epochs,
        verbose=mlp.verbosity,
        callbacks=callbacks,
        validation_split=mlp.split_validation,
    )
    end = time.time()
    runtime = (end - start) / 60  # in minutes

    # evaluate mlp applying test set
    score = model.evaluate(X_test, y_test, verbose=0)

    output_results(
        output_path,
        modeltype,
        mlp,
        features_dict,
        runtime,
        incident_types,
        score=score,
        history=history,
    )

    if best_model:
        best_model.update_best_model_info(
            score[1], settings, model, X_train, y_train, X_test, y_test, features_dict
        )

    return score[1]
