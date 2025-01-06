from typing import Any, Dict, List, Optional, Union

import yaml
from Bayesian_Optimization.bayesian_optimization import BO_with_Dropout
from Bayesian_Optimization.helper import read_space
from Data_Handler.data_storage import DataStorage
from Models.cnn_generator import run_CNN
from Models.mlp_generator import run_MLP
from Models.types import DataType, ModelType
from skopt import Space, dummy_minimize, forest_minimize, gp_minimize
from skopt.space import Categorical, Integer, Real


# pylint: disable=too-many-arguments
class BO_Handler:
    def __init__(
        self,
        data_storage,
        best_model,
        space_csv,
        settings_yaml,
        total_look_back,
        time_intervals,
        incident_types,
        nrRows,
        nrCols,
        output_path,
    ):
        self.data_storage = data_storage
        self.best_model = best_model
        self.space_csv = space_csv
        self.settings_yaml = settings_yaml
        self.total_look_back = total_look_back
        self.time_intervals = time_intervals
        self.incident_types = incident_types
        self.nrRows = nrRows
        self.nrCols = nrCols
        self.output_path = output_path

    def generate_search_space(
        self, space_csv: str, data_storage: DataStorage, optimize_features: bool = True
    ) -> List[Union[Integer, Real, Categorical]]:
        """Reads space settings from csv and adds space dimensions
        for feature selection if optimize_features = True

        Parameters:
        space_csv: str
            Name of csv file in which space configurations are saved
                Columns:
                    - name: name of dimensions (e.g. batch_size)
                    - type: Categorical, Integer or Real
                    - categories: If type is Categorical, provide all possible values (comma-separated)
                    - lower_bound: If type is Integer or Real, provide lower bound
                    - upper_bound: If type is Integer or Real, provide upper bound
                    - transform: transformation settings, e.g. normalize
        data_storage: DataStorage
            Class storing training data and feature information
        optimize_features: bool, default: True
            If true, for each feature to be included a boolean variable is added to the
            search space to represent the decision whether to include the corresponding feature or not

        Returns:
        space: List[Union[Integer, Real, Categorical]]
            List with space dimensions (Real, Categorical and Integer)
        """

        space = read_space(space_csv)

        # add binary variables for feature selection decision
        if optimize_features:

            for f in data_storage.get_features_overview(False):
                space.extend([Integer(0, 1, name=f.name)])

        return space

    def overwrite_settings_dict(
        self, settings_bays_opt: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates settings dictionary (key: name of setting variable, value: settings value)
        Values are read from <self.settings_yaml> and are overwritten by values in <settings_bays_opt>

        Parameters:
        settings_bays_opt: Dict[str, Any]
            Dictionary including parameter names and values used for overwriting old values
            Keys: name of setting variable, values: settings values

        Returns:
        settings_dict: Dict[str, Any]
            Dictionary including parameter names and values.
            All values which are varied by BO are overwritten.
            Remaining parameter values remain (same as in <self.settings_yaml>)
            Keys: name of setting variable, values: settings values
        """
        # use fixed parameters as basis
        with open(self.settings_yaml, "r") as stream:
            try:
                settings_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # get parameters that are varied by bayesian optimization
        feature_names_space_bayes_opt = Space(
            read_space(self.space_csv)
        ).dimension_names

        # overwrite the default parameter values with the new promising values obtained by BO
        # i.e. in each iteration of the BO, the parameters are overwritten
        for f in feature_names_space_bayes_opt:
            settings_dict[f] = settings_bays_opt[f]

        return settings_dict

    def min_CNN(self, arr: List[Any]) -> float:
        """
        Runs CNN for array <arr> containing all settings and features, feature selection included.

        Parameters:
        arr: List[Any]
            List of values suggested by Bayesian Optimization
            List includes binary variables for feature selection decisions

        Returns:
        score: float
            Metric for measuring performance of CNN
        """
        # get names and "new" values of features (varied in BO)
        # and save in <settings_bays_opt>
        features_names_bayes_opt = Space(read_space(self.space_csv)).dimension_names
        features_values_bayes_opt = arr[: len(features_names_bayes_opt)]

        settings_bays_opt = {}
        for idx, f in enumerate(features_names_bayes_opt):
            settings_bays_opt[f] = features_values_bayes_opt[idx]

        # overwrite default parameters with varied parameters
        settings_dict = self.overwrite_settings_dict(settings_bays_opt)

        # all remaining elements must be feature settings
        # (i.e. binary variables for feature selection)
        feature_array = arr[len(features_names_bayes_opt) :]

        # update data set with feature settings and run CNN
        self.data_storage.update_data_set(
            filter_array=feature_array, data_types_to_update=DataType.LAYER_BASED
        )
        score = run_CNN(
            settings_dict,
            self.data_storage,
            self.total_look_back,
            self.incident_types,
            self.nrRows,
            self.nrCols,
            best_model=self.best_model,
            output_path=self.output_path,
        )

        return score

    def min_CNN_no_feature_opt(self, arr: List[Any]) -> float:
        """
        Runs CNN for array with settings (no feature selection included).

        Parameters:
        arr: List[Any]
            List of values suggested by Bayesian Optimization
            List DOES NOT include binary variables for feature selection decisions

        Returns:
        score: float
            Metric for measuring the model's performance
        """

        # get names and "new" values of features (varied in BO)
        # and save in <settings_bays_opt>
        features_names_bayes_opt = Space(read_space(self.space_csv)).dimension_names

        settings_bays_opt = {}
        idx = 0
        for f in features_names_bayes_opt:
            settings_bays_opt[f] = arr[idx]
            idx += 1

        # overwrite default parameters with varied parameters
        settings_dict = self.overwrite_settings_dict(settings_bays_opt)

        # update data set with feature settings and run model
        self.data_storage.update_data_set(
            filter_array=self.data_storage.filter_array,
            data_types_to_update=DataType.LAYER_BASED,
        )
        score = run_CNN(
            settings_dict,
            self.data_storage,
            self.total_look_back,
            self.incident_types,
            self.nrRows,
            self.nrCols,
            best_model=self.best_model,
            output_path=self.output_path,
        )

        return score

    def min_MLP(self, arr: List[Any]) -> float:
        """
        Runs MLP for array with settings and features, feature selection included.

        Parameters:
        arr: List[Any]
            List of values suggested by Bayesian Optimization
            List includes binary variables for feature selection decisions

        Returns:
        score: float
            Metric for measuring the model's performance
        """

        # get names and "new" values of features (varied in BO)
        # and save in <settings_bays_opt>
        features_names_bayes_opt = Space(read_space(self.space_csv)).dimension_names
        features_values_bayes_opt = arr[: len(features_names_bayes_opt)]

        settings_bays_opt = {}
        idx = 0
        for f in features_names_bayes_opt:
            settings_bays_opt[f] = features_values_bayes_opt[idx]
            idx += 1

        # overwrite default parameters with varied parameters
        settings_dict = self.overwrite_settings_dict(settings_bays_opt)

        # all remaining elements must be feature settings
        # (i.e. binary variables for feature selection)
        feature_array = arr[len(features_names_bayes_opt) :]

        # update data set with feature settings and run model
        self.data_storage.update_data_set(
            filter_array=feature_array, data_types_to_update=DataType.INSTANCE_BASED
        )
        return run_MLP(
            settings_dict,
            self.data_storage,
            self.incident_types,
            best_model=self.best_model,
            output_path=self.output_path,
        )

    def min_MLP_no_feature_opt(self, arr: List[Any]) -> float:
        """
        Runs CNN for array with settings (no feature selection included).

        Parameters:
        arr: List[Any]
            List of values suggested by Bayesian Optimization
            List DOES NOT include binary variables for feature selection decisions

        Returns:
        score: float
            Metric for measuring the model's performance
        """

        # get names and "new" values of features (varied in BO)
        # and save in <settings_bays_opt>
        features_names_bayes_opt = Space(read_space(self.space_csv)).dimension_names

        settings_bays_opt = {}
        idx = 0
        for f in features_names_bayes_opt:
            settings_bays_opt[f] = arr[idx]
            idx += 1

        # overwrite default parameters with varied parameters
        settings_dict = self.overwrite_settings_dict(settings_bays_opt)

        # update data set with feature settings and run model
        self.data_storage.update_data_set(
            filter_array=self.data_storage.filter_array,
            data_types_to_update=DataType.INSTANCE_BASED,
        )
        return run_MLP(
            settings_dict,
            self.data_storage,
            self.incident_types,
            best_model=self.best_model,
            output_path=self.output_path,
        )

    # pylint: disable=too-many-arguments
    def run_bayesian_optimization(
        self,
        modeltype: ModelType,
        prior: str,
        nr_calls: int = 500,
        n_initial_points: Optional[int] = None,
        n_initial_points_percentage: float = 0.1,
        optimize_features: bool = True,
        bayesian_dropout: bool = False,
        p: int = 0,
        d: int = 0,
        x0: Optional[List[List[Any]]] = None,
        y0: Optional[List[float]] = None,
        random_state: Optional[int] = None,
        acq_func: str = "EI",
    ):
        """
        Runs Bayesian Optimization.

        Parameters:
            modeltype: Modeltype
                Supported: ModelType.MLP, ModelType.CNN

            prior: str
                - "RF" Random Forest
                - "RS" Random Search
                - "ET" Extremely Randomized Trees
                - "GP" Gausian Processes

            nr_calls : int, default: 100
                Number of calls to `func`.

            n_initial_points : int, default: None
                Number of evaluations of `func` with initialization points
                before approximating it with `base_estimator`. Initial point
                generator can be changed by setting `initial_point_generator`.
                If "None", n_initial_points_percentage used to calculate n_initial_points

            n_initial_points_percentage: int, default: 0.1
                If n_initial_points set to "None", n_initial_points_percentage used to calculate n_initial_points by
                multiplying n_initial_points_percentage with nr_calls

            optimize_features: bool, default: True
                True, if features should be optimized

            bayesian_dropout: bool, default: False
                True, if bayesian optimizatio with dimension dropout should be applied

            p: float, default: 0
                Probability to apply incumbent parameter values for fill-up parameters when applying bayesian optimization with dimensian dropout
                (1-p): Probability to fill-up parameter values randomly
                Only applied if bayesian_dropout = True

            d: int, default: 0
                Number of dimensions that should be randomly drawn (and optimized) when conducting dimension dropout
                Only applied if bayesian_dropout = True

            x0 : list, list of lists or `None`
                Initial input points.

                - If it is a list of lists, use it as a list of input points.
                - If it is a list, use it as a single initial input point.
                - If it is `None`, no initial input points are used.

            y0 : list, scalar or `None`
                Evaluation of initial input points.

                - If it is a list, then it corresponds to evaluations of the function
                at each element of `x0` : the i-th element of `y0` corresponds
                to the function evaluated at the i-th element of `x0`.
                - If it is a scalar, then it corresponds to the evaluation of the
                function at `x0`.
                - If it is None and `x0` is provided, then the function is evaluated
                at each element of `x0`.

            random_state : int, RandomState instance, or None (default)
                Set random state to something other than None for reproducible
                results.

            acq_func : string, default: `"gp_hedge"`
                Function to minimize over the gaussian prior. Can be either

                - `"LCB"` for lower confidence bound.
                - `"EI"` for negative expected improvement.
                - `"PI"` for negative probability of improvement.
                - `"gp_hedge"` Probabilistically choose one of the above three
                acquisition functions at every iteration. The weightage
                given to these gains can be set by :math:`\\eta` through
                `acq_func_kwargs`.

                - The gains `g_i` are initialized to zero.
                - At every iteration,

                    - Each acquisition function is optimised independently to
                    propose an candidate point `X_i`.
                    - Out of all these candidate points, the next point `X_best` is
                    chosen by :math:`softmax(\\eta g_i)`
                    - After fitting the surrogate model with `(X_best, y_best)`,
                    the gains are updated such that :math:`g_i -= \\mu(X_i)`

                - `"EIps"` for negated expected improvement per second to take into
                account the function compute time. Then, the objective function is
                assumed to return two values, the first being the objective value and
                the second being the time taken in seconds.
                - `"PIps"` for negated probability of improvement per second. The
                return type of the objective function is assumed to be similar to
                that of `"EIps"`

        """

        # default: 10% of number of calls
        if n_initial_points is None:
            n_initial_points = int(nr_calls * n_initial_points_percentage)

        space = self.generate_search_space(
            self.space_csv, self.data_storage, optimize_features
        )

        # use custom_gp_minimize for Bayesian Optimization with Dimension Dropout
        # and set settings correspondingly, otherwise use skopt.gp_minimize
        if bayesian_dropout & (prior == "GP"):
            bo_with_dropout = BO_with_Dropout(p, int(d * len(space)), len(space))
            pr = bo_with_dropout.custom_gp_minimize
        elif prior == "GP":
            pr = gp_minimize

        # set function called by BO dependent on modeltype and feature selection settings
        if modeltype == ModelType.MLP:
            if optimize_features:
                func = self.min_MLP
            else:
                func = self.min_MLP_no_feature_opt
        elif modeltype == ModelType.CNN:
            if optimize_features:
                func = self.min_CNN
            else:
                func = self.min_CNN_no_feature_opt

        res = None

        # random search
        if prior == "RS":
            res = dummy_minimize(
                func,
                space,
                n_calls=nr_calls,
                x0=x0,
                y0=y0,
                random_state=random_state,
            )

        # random forest
        elif prior == "RF":
            res = forest_minimize(
                func,
                space,
                acq_func=acq_func,
                n_calls=nr_calls,
                n_initial_points=n_initial_points,
                random_state=random_state,
                x0=x0,
                y0=y0,
                base_estimator="RF",
            )

        # gaussian process
        elif prior == "GP":
            res = pr(
                func,
                space,
                acq_func=acq_func,
                n_calls=nr_calls,
                n_initial_points=n_initial_points,
                base_estimator="GP",
                x0=x0,
                y0=y0,
                random_state=random_state,
            )

        # extremely randomized trees
        elif prior == "ET":
            res = forest_minimize(
                func,
                space,
                acq_func=acq_func,
                n_calls=nr_calls,
                n_initial_points=n_initial_points,
                random_state=random_state,
                x0=x0,
                y0=y0,
                base_estimator="ET",
            )

        # output results
        return res
