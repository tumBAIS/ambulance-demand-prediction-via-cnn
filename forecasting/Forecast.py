# own modules
# external modules
import argparse
import os
import sys

import pandas as pd
import yaml
from Bayesian_Optimization.bayesian_optimization_handler import BO_Handler
from Bayesian_Optimization.bayesian_optimization_hierarchical import \
    run_hierarchical_bo
from Bayesian_Optimization.helper import read_x0, read_y0
from Data_Handler.data_storage import get_data_storage
from Models.cnn_generator import run_CNN
from Models.medic import run_medic_method
from Models.mlp_generator import run_MLP
from Models.tree_generator import DecTree, RanForest
from Models.types import DataType, ModelType
from Results_Handler.results_handler import BestModel
from Shap.shap_creator import ShapExplainer

# setting paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--settings")
args = parser.parse_args()
main_settings_yaml = args.settings  # "./Settings/settings_main.yaml"

with open(main_settings_yaml, "r") as stream:
    try:
        main_settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

output_path = main_settings["output_path"]
output_data = main_settings["output_data"]

# data storage only required when running a model
if (
    main_settings["run_model"]
    or main_settings["grid_search_tree"]
    or main_settings["run_BO"]
    or main_settings["run_hierarchical_BO"]
    or main_settings["medic_method"]
):

    start = pd.Timestamp(main_settings["start"])
    end = pd.Timestamp(main_settings["end"])
    medic_earlierst_data = pd.Timestamp(main_settings["medic_earlierst_data"])
    total_look_back = main_settings["total_look_back"]
    time_interval = main_settings["time_interval"]
    nrRows = main_settings["nrRows"]
    nrCols = main_settings["nrCols"]
    calldata = main_settings["calldata"]
    min_lat = main_settings["min_lat"]
    max_lat = main_settings["max_lat"]
    min_lon = main_settings["min_lon"]
    max_lon = main_settings["max_lon"]
    best_model = BestModel(output_path)

    # Generate temporal and spatial intervals
    latitude_intervals = pd.interval_range(start=min_lat, end=max_lat, periods=nrRows)
    longitude_intervals = pd.interval_range(start=min_lon, end=max_lon, periods=nrCols)
    time_intervals = pd.interval_range(start=start, end=end, freq=time_interval)
    time_intervals_medic = pd.interval_range(
        start=medic_earlierst_data, end=end, freq=time_interval
    )

    incident_types = main_settings["incident_types"].split(
        ","
    )  # Separate types by comma
    filter_dict = {"Type": incident_types}
    features_csv = main_settings["features_csv"]

    data_storage = get_data_storage(
        main_settings["load_data_storage_path"],
        main_settings["save_data_storage_path"],
        total_look_back,
        time_intervals,
        time_intervals_medic,
        latitude_intervals,
        longitude_intervals,
        nrRows,
        nrCols,
        calldata,
        filter_dict,
        features_csv,
        main_settings["shorten_one_hot"],
        data_types=[DataType.LAYER_BASED, DataType.INSTANCE_BASED],
    )

modeltype = main_settings["model"]

if modeltype == "CNN":
    modeltype = ModelType.CNN
    datatype = DataType.LAYER_BASED
elif modeltype == "MLP":
    modeltype = ModelType.MLP
    datatype = DataType.INSTANCE_BASED
elif modeltype == "DecisionTree":
    modeltype = ModelType.DECISION_TREE
    datatype = DataType.INSTANCE_BASED
elif modeltype == "RandomForest":
    modeltype = ModelType.RANDOM_FOREST
    datatype = DataType.INSTANCE_BASED
else:
    raise ValueError(f"Modeltype: {modeltype} unknown.")

if main_settings["run_model"]:

    best_model.output_all_updates = True

    with open(main_settings["model_settings"], "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if modeltype == ModelType.CNN:
        run_CNN(
            settings=settings,
            data_storage=data_storage,
            total_look_back=total_look_back,
            incident_types=incident_types,
            nrRows=nrRows,
            nrCols=nrCols,
            best_model=best_model,
            output_path=output_path,
        )

    elif modeltype == ModelType.MLP:
        run_MLP(
            settings=settings,
            data_storage=data_storage,
            incident_types=incident_types,
            best_model=best_model,
            output_path=output_path,
        )

    elif modeltype == ModelType.DECISION_TREE:
        dectree = DecTree(**settings)
        dectree.run(
            best_model=best_model,
            data_storage=data_storage,
            incident_types=incident_types,
            output_path=output_path,
        )

    elif modeltype == ModelType.RANDOM_FOREST:
        ranforest = RanForest(**settings)
        ranforest.run(
            best_model=best_model,
            data_storage=data_storage,
            incident_types=incident_types,
            output_path=output_path,
        )

    best_model.output_best_model_info(output_data)

# Bayesian Optimization
if main_settings["run_BO"]:

    if modeltype not in {ModelType.CNN, ModelType.MLP}:
        raise ValueError("BO not implemented for tree-based models.")

    best_model.output_all_updates = True

    if main_settings["x0"] is not None and main_settings["y0"] is not None:
        x0 = read_x0(main_settings["x0"])
        y0 = read_y0(main_settings["y0"])
    else:
        x0 = None
        y0 = None

    bo_handler = BO_Handler(
        data_storage,
        best_model,
        main_settings["space_csv"],
        main_settings["model_settings"],
        total_look_back,
        time_intervals,
        incident_types,
        nrRows,
        nrCols,
        output_path,
    )

    bo_handler.run_bayesian_optimization(
        modeltype=modeltype,
        prior=main_settings["prior"],
        nr_calls=main_settings["nr_calls"],
        optimize_features=main_settings["optimize_features"],
        n_initial_points=main_settings["n_initial_points"],
        bayesian_dropout=main_settings["bayesian_dropout"],
        p=main_settings["p"],
        d=main_settings["d"],
        x0=x0,
        y0=y0,
    )

    best_model.output_best_model_info(output_data)

# Hierarchical BO
if main_settings["run_hierarchical_BO"]:
    if modeltype not in {ModelType.CNN, ModelType.MLP}:
        raise ValueError("BO not implemented for tree-based models.")

    best_model.output_all_updates = True

    run_hierarchical_bo(
        output_path=output_path,
        total_look_back=total_look_back,
        nrRows=nrRows,
        nrCols=nrCols,
        best_model=best_model,
        time_intervals=time_intervals,
        incident_types=incident_types,
        data_storage=data_storage,
        modeltype=modeltype,
        datatype=datatype,
        space_csvs=main_settings["space_csv_hierarchical_BO"],
        priors=main_settings["priors_hierarchical_BO"],
        optimize_features_level=main_settings["optimize_features_level"],
        settings_yaml=main_settings["model_settings"],
        nr_calls=main_settings["nr_calls_hierarchical_BO"],
        n_initial_points=main_settings["n_initial_points_hierarchical_BO"],
    )

# Grid Search Random Forest / Decision Tree
if main_settings["grid_search_tree"]:

    # settings to be optimized will be overwritten (e.g. max_depth, n_estimators)
    with open(main_settings["model_settings"], "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if modeltype == ModelType.DECISION_TREE:
        dectree = DecTree(**settings)
        dectree.optimize(
            data_storage,
            incident_types,
            best_model,
            main_settings["max_depth_lower"],
            main_settings["max_depth_upper"],
            main_settings["max_depth_stepsize"],
            output_path=output_path,
        )

    elif modeltype == ModelType.RANDOM_FOREST:
        ranforest = RanForest(**settings)
        ranforest.optimize(
            data_storage,
            incident_types,
            best_model,
            main_settings["max_depth_lower"],
            main_settings["max_depth_upper"],
            main_settings["max_depth_stepsize"],
            main_settings["n_estimators_lower"],
            main_settings["n_estimators_upper"],
            main_settings["n_estimators_setpsize"],
            output_path=output_path,
        )

    else:
        raise ValueError(
            f"Grid_search_tree set True, but {modeltype} selected. Not a tree-based model."
        )

    best_model.output_best_model_info(output_data)

if main_settings["medic_method"]:

    run_medic_method(
        data_storage,
        time_intervals,
        time_intervals_medic,
        main_settings["nr_weeks_to_include"],
        main_settings["nr_years_to_include"],
        nrRows,
        nrCols,
        time_interval,
        output_path=output_path,
    )

# SHAP values calculation
if main_settings["calculate_shap_values"]:

    ShapExplainer().calculate_shap_values(
        model_path=main_settings["model_path"],
        X_train_path=main_settings["X_train_path"],
        X_test_path=main_settings["X_test_path"],
        features_dict_path=main_settings["features_overview_path"],
        shap_explainer=main_settings["explainer"],
        output_path=output_path + "/Shap/",
        ninstances_explain=main_settings["ninstances_explain"],
        ninstances_train=main_settings["ninstances_train"],
        nsamples=main_settings["nsamples"],
        multiplier=main_settings["multiplier"],
        check_additivity=main_settings["check_additivity"],
    )
