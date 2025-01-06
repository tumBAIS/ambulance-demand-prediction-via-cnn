import csv
import os.path
import pickle
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from Bayesian_Optimization.helper import replace_objects
from Data_Handler.feature_names import Feature_w_Time
from Models.types import ModelType
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import models


def output_results(
    output_path: str,
    modeltype: ModelType,
    model: models.Model,
    features: Dict[Feature_w_Time, List[str]],
    runtime: float,
    incident_types: List[str],
    history: Any = None,
    score: Any = None,
    metrics_trees: Any = None,
) -> None:
    """Dumps results to specified <output_path>

    Parameters:
    output_path: str
        Path where results are dumped
    modeltype: ModelType
        Type of model
    model: tensorflow.keras.models.Model
        trained model
    features: Dict[Feature_w_Time, List[str]]
        list of features applied
    runtime: float
        algorithm runtime
    incident_types: List[str]
        list of incident types
    history: Any, Optional
        historical training data
    score: Any, Optional
        value of metric applied to evaluate model performance
    metrics_trees: Any, Optional
        metrics applied for trees
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    headers = model.data_headers
    results = model.data
    headers.extend(["runtime", "features", "incident_types"])
    results.extend(
        [
            runtime,
            [
                feature.name
                for feature in features
                if feature != Feature_w_Time.AMBULANCE_DEMAND_HISTORY
            ],
            incident_types,
        ]
    )

    if modeltype in (ModelType.MLP, ModelType.CNN):
        # If model is any kind of neural network (CNN, MLP)
        if modeltype == ModelType.CNN:
            results_csv_file = f"{output_path}/results_neuralnets_CNN.csv"
            hist_csv_file = f"{output_path}/history_CNN.csv"
        elif modeltype == ModelType.MLP:
            results_csv_file = f"{output_path}/results_neuralnets_MLP.csv"
            hist_csv_file = f"{output_path}/history_MLP.csv"

        # headers
        headers.extend([*model.model.metrics_names])
        results.extend([*score])

        # Print learning history to csv (only applicable to neural nets)
        hist_df = pd.DataFrame(history.history)
        hist_df["Run ID"] = model.id
        hist_csv_file_exists = os.path.isfile(hist_csv_file)
        with open(hist_csv_file, mode="a") as f:
            hist_df.to_csv(f, header=(not hist_csv_file_exists))

        # Save weights
        # weight = model.model.get_weights()
        # np.savetxt('Results/Weights/Weights_' + str(model.id) + '.csv' , weight , fmt='%s', delimiter=',')

    # If model is any kind of tree (Decision Tree, Random Forest)
    if modeltype in (ModelType.DECISION_TREE, ModelType.RANDOM_FOREST):
        headers.extend([*list(metrics_trees.keys())])
        results.extend([*list(metrics_trees.values())])

        if modeltype == ModelType.DECISION_TREE:
            results_csv_file = f"{output_path}/results_dectrees.csv"

        elif modeltype == ModelType.RANDOM_FOREST:
            results_csv_file = f"{output_path}/results_ranforests.csv"

    # convert results to df and replace "strange" object names by nice string names (e.g. layers.keras...ReLu -> relu)
    results = generate_readable_results(results, headers)

    # Print results to corresponding file
    results_csv_file_exists = os.path.isfile(results_csv_file)

    with open(results_csv_file, mode="a") as f:
        writer = csv.writer(f)
        if not results_csv_file_exists:
            writer.writerow(headers)
        writer.writerow(*results)


def generate_readable_results(results: List[Any], headers: List[str]):
    """Transforms object names to better readable strings (eg. tf.keras.losses.MSE -> MSE)

    Parameters:
    results: list
        List of lists with results to be printed with not nicely readable elements
        Each nested list includes all parameters and results from one results instance
    headers: list
        List with headers matching the contents of results list

    Returns:
    list
        List of lists with results to be printed with nicely readable elements.
    """
    if not isinstance(results[0], list):
        results = [results]
    results_df = pd.DataFrame(data=results, columns=headers)
    results_df = replace_objects(results_df)
    return results_df.values.tolist()


# pylint: disable=too-many-instance-attributes
class BestModel:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.output_all_updates = False
        self.best_settings = None
        self.best_model = None
        self.best_func_value = float("inf")
        self.best_model = None
        self.best_X_train = None
        self.best_y_train = None
        self.best_X_test = None
        self.best_y_test = None
        self.best_feature_names = None
        self.best_feature_overview = None
        self.best_feature_importance = None
        self.lock = Lock()

    def update_best_model_info(
        self,
        func_value: float,
        settings: Dict[str, Any],
        model: models.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        best_features_dict: Dict[Feature_w_Time, List[str]],
        feature_importance=None,
    ):
        """Stores incumbent parameter settings and data sets

        Parameters:
        func_value: float
            objective value
        settings: Dict[str, Any]
            parameter settings
        model: tensorflow.keras.models.Model
            trained model
        X_train: np.ndarray
            training data, independent variables
        y_train: np.ndarray
            training data, dependent variables
        X_test: np.ndarray
            test data, independent variables
        y_test: np.ndarray
            test data, dependent variables
        best_features_dict: Dict[Feature_w_Time, List[str]]
            list of features applied
        feature_importance: Optional
            feature importances (calculated only for trees)
        """
        self.lock.acquire()
        if func_value < self.best_func_value:
            self.best_func_value = func_value
            self.best_settings = settings
            self.best_model = model
            self.best_X_train = X_train
            self.best_y_train = y_train
            self.best_X_test = X_test
            self.best_y_test = y_test
            self.best_features_dict = best_features_dict
            self.best_feature_importance = feature_importance
            if self.output_all_updates:
                self.output_best_model_info()
        self.lock.release()

    def output_best_model_info(
        self, output_data: bool = False, alternative_output_path: Optional[str] = None
    ) -> None:
        """Dumps results to specified path

        Parameters:
        output_data: bool
            If True, datasets (training and test instances) are dumped
        alternative_output_path: Optional[str]
            If given, default output_path is overwritten by this path
        """

        if alternative_output_path is None:
            output_path = self.output_path
        else:
            output_path = alternative_output_path

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not (
            isinstance(
                self.best_model, (tree.DecisionTreeRegressor, RandomForestRegressor)
            )
        ):
            self.best_model.save(f"{output_path}/model.h5")
        else:
            with open(f"{output_path}/model.p", "wb") as file:
                pickle.dump(self.best_model, file)

        if output_data:
            with open(f"{output_path}/X_test.p", "wb") as file:
                pickle.dump(self.best_X_test, file)

            with open(f"{output_path}/X_train.p", "wb") as file:
                pickle.dump(self.best_X_train, file)

            with open(f"{output_path}/y_test.p", "wb") as file:
                pickle.dump(self.best_y_test, file)

            with open(f"{output_path}/y_train.p", "wb") as file:
                pickle.dump(self.best_y_train, file)

        if isinstance(
            self.best_model, (RandomForestRegressor, tree.DecisionTreeRegressor)
        ):
            with open(f"{output_path}/self.best_feature_importance.p", "wb") as file:
                pickle.dump(self.best_feature_importance, file)

        with open(f"{output_path}/features_dict.p", "wb") as file:
            pickle.dump(self.best_features_dict, file)

        with open(f"{output_path}/final_results.txt", "w") as f:
            f.write(f"{self.best_func_value},\n")
            f.write(f"{self.best_settings}\n")
            f.write(f"{[f.name for f in self.best_features_dict]}")
