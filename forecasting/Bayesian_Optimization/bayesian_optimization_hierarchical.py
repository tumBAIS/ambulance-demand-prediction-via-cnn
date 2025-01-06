from copy import deepcopy
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from Bayesian_Optimization.bayesian_optimization_handler import BO_Handler
from Data_Handler.data_storage import DataStorage
from Models.types import DataType, ModelType
from Results_Handler.results_handler import BestModel


# Hierarchical Bayesian Optimization
def run_hierarchical_bo(
    output_path: str,
    total_look_back: int,
    nrRows: int,
    nrCols: int,
    best_model: BestModel,
    time_intervals: List[pd.Interval],
    incident_types,
    data_storage: DataStorage,
    modeltype: ModelType,
    datatype: DataType,
    space_csvs: List[str],
    priors: List[str],
    optimize_features_level: Optional[int],
    settings_yaml: str,
    nr_calls: int,
    n_initial_points: int,
) -> None:
    """
    Runs hierarchical Bayesian optimization

    Parameters:
    output_path: str
        Path where to store results
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    nrRows: int
        number of horizontal area divisions (=len(longitude_intervals))
    nrCols: int
        number of vertical area divisions (=len(latitude_intervals))
    best_model: BestModel
        BestModel instance to save incumbent settings
    incident_types,
    data_storage: DataStorage
        DataStorage instance with saved data for provided models
    modeltype: ModelType
        ModelType of model run by hierarchical BO
    datatype: DataType
        DataType used for Model
    space_csvs: List[str]
        List with csv file paths which contain space settings
    priors: List[str]
        List of priors (supported: "RS", "GP", "ET", "RF")
    optimize_features_level: Optional[int]
        Level at which features should be optimized (normally 0)
    settings_yaml: str
        Path to file in which initial settings are stored
    nr_calls: int
        Number of BO iterations
    n_initial_points: int
        Number of initial points for BO
    """

    incumbent_fun = float("inf")

    for level, space_csv in enumerate(space_csvs):
        space_df = pd.read_csv(space_csv, delimiter=";")
        hyperparameter_names = space_df["name"].tolist()
        optimize_features = level == optimize_features_level
        update_hyperparameters = False

        results_dict = {}
        jobs = [
            Thread(
                target=__run_HBO_for_prior,
                args=(
                    output_path,
                    total_look_back,
                    nrRows,
                    nrCols,
                    best_model,
                    time_intervals,
                    incident_types,
                    data_storage,
                    modeltype,
                    settings_yaml,
                    nr_calls,
                    n_initial_points,
                    level,
                    space_csv,
                    optimize_features,
                    prior,
                    results_dict,
                ),
            )
            for prior in priors
        ]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        for prior in priors:
            # if better hyperparameter setting found, update incumbent values
            if results_dict[prior]["fun"] < incumbent_fun:
                incumbent_prior = prior
                incumbent_fun = results_dict[prior]["fun"]
                incumbent_hyperparameter_values = [
                    x.item() if hasattr(x, "dtype") else x
                    for x in results_dict[prior]["x"]
                ]
                update_hyperparameters = True

        # save incumbent hyperparameter settings and overwrite settings applied
        # in following iterations
        if update_hyperparameters:

            # save incumbent parameter values used to overwrite in dict
            updated_hyperparameters = {}
            hyperparameter_values = incumbent_hyperparameter_values[
                : len(hyperparameter_names)
            ]
            for idx, name in enumerate(hyperparameter_names):
                updated_hyperparameters[name] = hyperparameter_values[idx]

            # read "old" hyperparameter settings
            with open(settings_yaml, "r") as stream:
                try:
                    settings_dict = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # update "old" hyperparameter settings (excl. feature selection)
            # update them and save
            settings_dict.update(updated_hyperparameters)

            settings_yaml = settings_yaml.replace(
                ".yaml", f"_{incumbent_prior}_L{level}.yaml"
            )

            with open(settings_yaml, "w") as file:
                yaml.dump(settings_dict, file)

            # update selected features in data storage
            if optimize_features:
                filter_array = incumbent_hyperparameter_values[
                    len(hyperparameter_names) :
                ]
                data_storage.update_data_set(
                    filter_array=filter_array, data_types_to_update=datatype
                )


def __run_HBO_for_prior(
    output_path: str,
    total_look_back: int,
    nrRows: int,
    nrCols: int,
    best_model: BestModel,
    time_intervals: List[pd.Interval],
    incident_types,
    data_storage: DataStorage,
    modeltype: ModelType,
    settings_yaml: str,
    nr_calls: int,
    n_initial_points: int,
    level: int,
    space_csv: str,
    optimize_features: bool,
    prior: str,
    results_dict: Dict[Any, Any],
) -> None:
    """
    Triggers BO and outputs incumbent results

    Parameters:
    output_path: str
        Path where to store results
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    nrRows: int
        number of horizontal area divisions (=len(longitude_intervals))
    nrCols: int
        number of vertical area divisions (=len(latitude_intervals))
    best_model: BestModel
        BestModel instance to save incumbent settings
    incident_types,
    data_storage: DataStorage
        DataStorage instance with saved data for provided models
    modeltype: ModelType
        ModelType of model run by hierarchical BO
    settings_yaml: str
        Path to file in which initial settings are stored
    nr_calls: int
        Number of BO iterations
    n_initial_points: int
        Number of initial points for BO
    level: int
        Hierarchical level of current run
    space_csv: str
        Name of csv file in which space configurations are saved
            Columns:
                - name: name of dimensions (e.g. batch_size)
                - type: Categorical, Integer or Real
                - categories: If type is Categorical, provide all possible values (comma-separated)
                - lower_bound: If type is Integer or Real, provide lower bound
                - upper_bound: If type is Integer or Real, provide upper bound
                - transform: transformation settings, e.g. normalize
    optimize_features: bool
        If True, feature selection is included in hyperparameter tuning process
    prior: str
        prior to be used (supported: "RS", "GP", "ET", "RF")
    results_dict: Dict[Any, Any]
        dictionary to store results

    """
    bo_handler = BO_Handler(
        deepcopy(data_storage),
        best_model,
        space_csv,
        settings_yaml,
        total_look_back,
        time_intervals,
        incident_types,
        nrRows,
        nrCols,
        f"{output_path}/{prior}_L{level}",
    )

    results = bo_handler.run_bayesian_optimization(
        modeltype=modeltype,
        prior=prior,
        nr_calls=nr_calls,
        optimize_features=optimize_features,
        n_initial_points=n_initial_points,
    )

    with open(f"{output_path}/{prior}_L{level}/results.txt", "w") as f:
        f.write(f'{results["fun"]}\n')
        f.write(f'{[x.item() if hasattr(x, "dtype") else x for x in results["x"]]}')

    results_dict[prior] = results
