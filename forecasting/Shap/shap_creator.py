# pylint: disable=invalid-name
import math
import os
import pickle
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from Data_Handler.feature_names import Feature_w_Time
from Shap.shap_adapter import adapt_inputs, flatten_nested_list
from tensorflow.keras import models

tf.compat.v1.disable_eager_execution()

SHAP_OBJ_OUTPUT_NAME = "shap_obj.p"
SHAP_VALUES_DF_OUTPUT_NAME = "shap_values_df.p"
SHAP_VALUES_PICKLE_OUTPUT_NAME = "shap_values.p"


class ShapExplainerType(Enum):
    """Implemented SHAP-Explainer Types"""

    SAMPLING = "sampling"
    TREE = "tree"


class ShapExplainer:
    def calculate_shap_values(
        self,
        model_path: str,
        X_train_path: str,
        X_test_path: str,
        features_dict_path: str,
        shap_explainer: str,
        output_path: str,
        ninstances_explain: Optional[int] = None,
        ninstances_train: Optional[int] = None,
        nsamples: Union[str, int] = "auto",
        multiplier: int = 1,
        check_additivity: bool = True,
    ) -> None:
        """Calculates and outputs SHAP values

        Parameters:
        model_path: str
            Path to model (supported file types: .h5 or .p)
        X_train_path: str
            Path to training data
        X_test_path: str
            Path to test data
        features_overview_path: str
            Path to feature names data
        shap_explainer: str
            Name of shap explainer, supported:
            - 'sampling'
            - 'tree'
        output_path: str
            Path at which outputs should be saved
        ninstances_explain: Optional[int], default: None
            Number of instances to explain
            If None, all instances of test data used
        ninstances_train: Optional[int], default: None
            Number of instances to train
            If None, all instances of training data used
        nsamples: Union[str, int], default "auto"
            Number of samples
        multiplier: int
            Nr. of subregions in data set (to scale data set for iterative approaches)
        check_additivity: bool
            Ignored for 'sampling' explainer, only applied to 'tree' explainer
            Run a validation check that the sum of the SHAP values equals the output of the model.
            This check takes only a small amount of time, and will catch potential unforeseen errors.
            Note that this check only runs right now when explaining the margin of the model.
        """

        # load data
        (model, X_train, X_test, shap_explainer, features_dict,) = self.__load_settings(
            model_path,
            X_train_path,
            X_test_path,
            features_dict_path,
            shap_explainer,
        )

        model, X_train_instances, X_test_instances = adapt_inputs(
            model, X_train, ninstances_train, X_test, ninstances_explain, multiplier
        )

        # calculate shap values
        shap_obj, shap_values = self.__calculate_shap_values(
            shap_explainer,
            model,
            features_dict,
            X_train_instances,
            X_test_instances,
            nsamples,
            check_additivity,
        )

        self.__output_results(output_path, shap_obj, shap_values, features_dict)

    def generate_shap_values_dataframe(
        self,
        shap_vals: np.ndarray,
        features_dict: Dict[Feature_w_Time, List[str]],
    ) -> pd.DataFrame:
        """Generates dataframe with SHAP values.
        Columns: Shap values, Feature name, Feature group, Square (only for CNN), Sample

        Parameters:
        shap_vals: np.ndarray
            Shap values to be added to dataframe
        features_dict: Dict[Feature_w_Time, List[str]]
                Feature overview
                - Key: Feature group (e.g. Feature_w_Time.HOURS_PREDICTION)
                - Values: e.g. ["Hour_0", "Hour_8", "Hour_16"]

        Returns:
         pd.DataFrame
            Dataframe with SHAP values assigned to each feature, square and sample
        """

        shap_values = shap_vals

        if len(shap_values.shape) == 2:
            shap_values = shap_values.reshape(
                shap_values.shape[0], shap_values.shape[1], 1
            )

        values = []
        feature_name = []
        feature_group = []
        square = []
        sample = []
        feature_names = flatten_nested_list(list(features_dict.values()))
        if len(feature_names) != shap_values.shape[1]:
            raise ValueError(
                f"Nr. of features {len(feature_names)} != {shap_values.shape[1]}! "
                "Check shap values shape and feature names (feature_dict.values())"
            )
        for s in range(shap_values.shape[0]):
            for f in range(shap_values.shape[1]):
                for sq in range(shap_values.shape[2]):
                    if not math.isnan(shap_values[s][f][sq]):
                        feature_group.append(
                            next(
                                feature_group.value
                                for feature_group in features_dict.keys()
                                if feature_names[f] in features_dict[feature_group]
                            )
                        )
                        values.append(shap_values[s][f][sq])
                        feature_name.append(feature_names[f])
                        sample.append(s)
                        # square only relevant for CNN (layer based)
                        if len(shap_vals.shape) == 2:
                            square.append(None)
                        else:
                            square.append(sq)

        return pd.DataFrame(
            {
                "Shap values": values,
                "Feature name": feature_name,
                "Feature group": feature_group,
                "Square": square,
                "Sample": sample,
            }
        )

    def __calculate_shap_values(
        self,
        shap_explainer: ShapExplainerType,
        model: models.Model,
        features_dict: Dict[Feature_w_Time, List[str]],
        X_train_instances: np.ndarray,
        X_test_instances: np.ndarray,
        nsamples: int,
        check_additivity: bool = True,
    ) -> Tuple[Optional[shap._explanation.Explanation], Optional[np.ndarray]]:
        """Calculates shap values

        Parameters
        shap_explainer: ShapExplainerType
            Shap Explainer to be used
        model: models.Model
            Model used to calculate SHAP values
        features_dict: Dict[Feature_w_Time, List[str]]
            Feature overview
            - Key: Feature group (e.g. Feature_w_Time.HOURS_PREDICTION)
            - Values: e.g. ["Hour_0", "Hour_8", "Hour_16"]
        X_train_instances: np.ndarray
            Training data to be used to calculate SHAP values (background data for sampling)
        X_test_instances: np.ndarray
            Test data to be explained
        nsamples: int,
            Number of samples for which SHAP values should be calculated
        check_additivity: bool
            Ignored for 'sampling' explainer, only applied to 'tree' explainer
            Run a validation check that the sum of the SHAP values equals the output of the model.
            This check takes only a small amount of time, and will catch potential unforeseen errors.
            Note that this check only runs right now when explaining the margin of the model.

        Returns:
        Optional[shap._explanation.Explanation]
            Shap explanation object containing shap values
        Optional[np.ndarray]
            Calculated shap values
        """

        shap_values = None
        shap_obj = None

        if shap_explainer == ShapExplainerType.SAMPLING:
            explainer = shap.SamplingExplainer(
                model=model.predict,
                features_overview=flatten_nested_list(list(features_dict.values())),
                data=X_train_instances,
            )
            shap_obj = explainer(X_test_instances, nsamples=nsamples)

        elif shap_explainer == ShapExplainerType.TREE:
            explainer = shap.TreeExplainer(model, X_train_instances)
            shap_values = explainer.shap_values(
                X_test_instances, check_additivity=check_additivity
            )

        return shap_obj, shap_values

    def __output_results(
        self,
        output_path: str,
        shap_obj: Optional[shap._explanation.Explanation],
        shap_values: Optional[np.ndarray],
        features_dict: Dict[Feature_w_Time, List[str]],
    ) -> None:
        """Calculates and outputs SHAP values

        Parameters:
        output_path: str
            Path at which outputs should be saved
        shap_obj: Optional[shap._explanation.Explanation]
            Explanation object containig shap values
        shap_values: Optional[np.ndarray]
            For models with a single output this is a matrix of SHAP values (# samples x # features).
            Each row sums to the difference between the model output for that sample and the expected value of the model output.
            For models with vector outputs this is a list of such matrices, one for each output.
        features_dict: Dict[Feature_w_Time, List[str]]
            Feature overview
            - Key: Feature group (e.g. Feature_w_Time.HOURS_PREDICTION)
            - Values: e.g. ["Hour_0", "Hour_8", "Hour_16"]
        """

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if shap_obj is not None:
            with open(output_path + SHAP_OBJ_OUTPUT_NAME, "wb") as file:
                pickle.dump(shap_obj, file)
            shap_df = self.generate_shap_values_dataframe(
                shap_obj.values, features_dict
            )

        if shap_values is not None:
            with open(output_path + SHAP_VALUES_PICKLE_OUTPUT_NAME, "wb") as file:
                pickle.dump(shap_values, file)
            shap_df = self.generate_shap_values_dataframe(shap_values, features_dict)

        with open(output_path + SHAP_VALUES_DF_OUTPUT_NAME, "wb") as file:
            pickle.dump(shap_df, file)

    def __load_settings(
        self,
        model_path: str,
        X_train_path: str,
        X_test_path: str,
        features_dict_path: str,
        shap_explainer: str,
    ) -> Tuple[
        models.Model,
        Union[np.ndarray, List[np.ndarray]],
        Union[np.ndarray, List[np.ndarray]],
        ShapExplainerType,
        List[str],
    ]:
        """Load data instances and settings from files

        Parameters:
        model_path: str
            Path to model (supported file types: .h5 or .p)
        X_train_path: str
            Path to training data
        X_test_path: str
            Path to test data
        features_dict_path: str
            Path to features dict
        shap_explainer: str
            Name of shap explainer, supported:
            - 'sampling'
            - 'tree'

        Returns:
        model: models.Model
            Model loaded from <model_path>
        X_train: Union[np.ndarray, List[np.ndarray]]
            Training data loaded from <X_train_path>
        X_test: Union[np.ndarray, List[np.ndarray]]
            Test data loaded from <X_train_path>
        shap_explainer: ShapExplainerType
            ShapExplainerTypes matching <shap_explainer>
        features_dict: Dict[Feature_w_Time, List[str]]
            Features dictionary loaded from <features_dict_path>
        """

        if model_path.endswith(".h5"):
            model = models.load_model(model_path, compile=False)
        elif model_path.endswith(".p"):
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
        else:
            raise ValueError(
                f"Unknown model data type: {model_path}. Supported: .h5 or .p"
            )

        with open(X_train_path, "rb") as X_train_file:
            X_train = pickle.load(X_train_file)
        with open(X_test_path, "rb") as X_test_file:
            X_test = pickle.load(X_test_file)
        with open(features_dict_path, "rb") as features_dict_file:
            features_dict = pickle.load(features_dict_file)

        if shap_explainer == "sampling":
            shap_explainer = ShapExplainerType.SAMPLING
        elif shap_explainer == "tree":
            shap_explainer = ShapExplainerType.TREE
        else:
            raise ValueError(f"Unknown shap explainger type: {shap_explainer}")

        return model, X_train, X_test, shap_explainer, features_dict
