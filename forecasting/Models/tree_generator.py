import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from Data_Handler.data_storage import DataStorage, generate_training_test_data
from Data_Handler.feature_names import Feature_w_Time
from Models.types import DataType, ModelType
from Results_Handler.results_handler import BestModel, output_results
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


# pylint: disable=no-member
class Tree:
    def __init__(self, **kwargs):
        self.id = str(datetime.today()).replace(":", "-")
        # Dynamically defined hyperparameters via Settings file (max_depth, n_estimator)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.data_type = DataType.INSTANCE_BASED

    def update_output_data(self):
        dic_attributes = self.__dict__.copy()
        del dic_attributes["model"]
        if "data" in dic_attributes:
            del dic_attributes["data"]
        if "data_headers" in dic_attributes:
            del dic_attributes["data_headers"]
        self.data = list(dic_attributes.values())
        self.data_headers = list(dic_attributes.keys())

    def build_model(self):
        pass

    def update_best_model(
        self,
        best_model: BestModel,
        metrics: Dict[str, Any],
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        features_dict: Dict[Feature_w_Time, List[str]],
    ):
        pass

    # pylint: disable=too-many-arguments
    def run(
        self,
        best_model: BestModel,
        data_storage: DataStorage,
        incident_types: Dict[str, List[str]],
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        output_path: str = "./Results",
    ) -> None:

        features_dict = data_storage.get_features_dict(self.data_type, True)

        # generate training and testset
        if X_train is None or y_train is None or X_test is None or y_test is None:

            # get data (if necessary, i.e. X any y not given)
            # check if data (X and y) was saved in data_storage, if yes, get this data
            if (
                (X is None or y is None)
                and (self.data_type in data_storage.X.keys())
                and (self.data_type in data_storage.y.keys())
            ):
                X = data_storage.X[self.data_type]  # already filtered
                y = data_storage.y[self.data_type]  # already filtered

            else:
                raise ValueError("Please provide X or y or store data in data_storage.")

            X_train, y_train, X_test, y_test = generate_training_test_data(
                *X,
                y=y,
                test_size=self.test_size,
                random_state=self.random_state,
                data_type=self.data_type,
                shuffle_data=self.shuffle_data,
            )

        # pylint: disable=assignment-from-no-return
        model = self.build_model()

        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        runtime = (end - start) / 60

        # make prediction and calculate MSE
        y_pred = model.predict(X_test)
        MSE = (1 / len(y_pred)) * sum(np.square(y_pred - y_test))

        y_pred_train = model.predict(X_train)
        MSE_train = (1 / len(y_pred)) * sum(np.square(y_pred_train - y_train))

        metrics = {"MSE": MSE, "MSE_train": MSE_train}

        self.update_output_data()
        self.update_best_model(
            best_model,
            metrics,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            features_dict,
        )

        output_results(
            output_path,
            self.modeltype,
            self,
            features_dict,
            runtime,
            incident_types,
            metrics_trees=metrics,
        )


class DecTree(Tree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modeltype = ModelType.DECISION_TREE

    def build_model(self):
        self.model = tree.DecisionTreeRegressor(
            criterion=self.criterion, max_depth=self.max_depth
        )
        return self.model

    # pylint: disable=too-many-arguments
    def optimize(
        self,
        data_storage: DataStorage,
        incident_types: Dict[str, List[str]],
        best_model: BestModel,
        max_depth_lower: int = 2,
        max_depth_upper: int = 100,
        max_depth_stepsize: int = 2,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        output_path: str = "./Results",
    ):

        for max_depth in range(max_depth_lower, max_depth_upper, max_depth_stepsize):

            # overwrite settings
            self.max_depth = max_depth

            super().run(
                best_model,
                data_storage,
                incident_types,
                X,
                y,
                X_train,
                y_train,
                X_test,
                y_test,
                output_path,
            )

    def update_best_model(
        self,
        best_model: BestModel,
        metrics: Dict[str, Any],
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        features_dict: Dict[Feature_w_Time, List[str]],
    ):
        best_model.update_best_model_info(
            metrics["MSE"],
            {"max_depth": self.max_depth},
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            features_dict,
            model.feature_importances_,
        )


# pylint: disable=no-member, too-many-arguments
class RanForest(Tree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.modeltype = ModelType.RANDOM_FOREST

    def build_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimator,
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state,
            oob_score=self.oob_score,
        )
        return self.model

    def optimize(
        self,
        data_storage,
        incident_types,
        best_model,
        max_depth_lower=2,
        max_depth_upper=4,
        max_depth_stepsize=2,
        n_estimators_lower=50,
        n_estimators_upper=100,
        n_estimators_setpsize=50,
        X=None,
        y=None,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        output_path: str = "./Results",
    ):

        for max_depth in range(max_depth_lower, max_depth_upper, max_depth_stepsize):
            for n_estimator in range(
                n_estimators_lower, n_estimators_upper, n_estimators_setpsize
            ):

                # overwrite settings
                self.max_depth = max_depth
                self.n_estimator = n_estimator

                # run model
                super().run(
                    best_model,
                    data_storage,
                    incident_types,
                    X,
                    y,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    output_path,
                )

    def update_best_model(
        self,
        best_model: BestModel,
        metrics: Dict[str, Any],
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        features_dict: Dict[Feature_w_Time, List[str]],
    ):
        best_model.update_best_model_info(
            metrics["MSE"],
            {"max_depth": self.max_depth, "n_estimator": self.n_estimator},
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            features_dict,
            model.feature_importances_,
        )
