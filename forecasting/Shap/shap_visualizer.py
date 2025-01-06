import os
import pickle
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# setting paths
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("..")

from Data_Handler.feature_names import Feature_w_Time


def generate_boxplots(df_dict: Dict[str, pd.DataFrame], scaling_factor: int):
    """Creates and saves shap value boxplots as png

    Parameters
    df_dict: Dict[str, pd.Dataframe]
        Key: Modelname, Value: Dataframe with column for each Feature group with corresponding shap values
    scaling_factor: int
        Scaling factor to scale shap values
    """

    data, features = __generate_boxplot_data(df_dict, scaling_factor)

    if len(data.keys()) > 3:
        raise NotImplementedError("Visualization only up to 3 models supported.")

    ticks = features

    colors = ["#e34a33", "#43a2ca", "#31a354"]

    if len(data.keys()) == 1:
        positions = [
            np.array(range(len(data[list(data.keys())[0]]))) * 2.0,
        ]
    elif len(data.keys()) == 2:
        positions = [
            np.array(range(len(data[list(data.keys())[0]]))) * 2.0 - 0.5,
            np.array(range(len(data[list(data.keys())[1]]))) * 2.0 + 0.5,
        ]
    elif len(data.keys()) == 3:
        positions = [
            np.array(range(len(data[list(data.keys())[0]]))) * 2.0 - 0.5,
            np.array(range(len(data[list(data.keys())[1]]))) * 2.0,
            np.array(range(len(data[list(data.keys())[2]]))) * 2.0 + 0.5,
        ]

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)
        plt.setp(bp["fliers"], color=color)

    plt.figure(figsize=(12, 12))
    plt.grid(color="black", linestyle="--", linewidth=0.1)

    for idx, model in enumerate(data.keys()):
        bp1 = plt.boxplot(
            data[model],
            positions=positions[idx],
            widths=1.2 / len(data.keys()),
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markeredgecolor": colors[idx],
                "markerfacecolor": colors[idx],
            },
        )
        set_box_color(bp1, colors[idx])
        plt.plot([], c=colors[idx], label=model)

    plt.legend(fontsize=16)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(-50, 80)
    plt.xlabel(xlabel="Feature", fontsize=16)
    plt.ylabel(ylabel=f"SHAP value [$x{scaling_factor}$]", fontsize=16)
    plt.tight_layout()
    plt.savefig("feature_importances.png")


def __generate_boxplot_data(
    df_dict: Dict[str, pd.DataFrame], scaling_factor: float
) -> Tuple[Dict[str, List[Any]], List[str]]:
    """Generates data required for shap value boxplots

    Parameters
    df_dict: dict
        - keys: str
            model name
        - values: pd.Dataframe
            Columns:
                - Shap values,
                - Feature name (e.g. Max. Temp [t-1]),
                - Feature group (e.g. Max. Temp),
                - Sample
    scaling_factor: float
        Factor with which shap values are multiplied before being plotted

    Returns
    data: Dict[str, List[Any]]
        Column for each Feature group with corresponding shap values
    features: List[str]
        List with Feature group names
    """

    dfs = list(df_dict.values())

    features = list(dict.fromkeys(dfs[0]["Feature group"]))

    for idx in range(len(dfs) - 1):
        if set(dfs[idx]["Feature group"]) != set(dfs[idx + 1]["Feature group"]):
            raise ValueError("Dataframes must have similar features!")

    data = defaultdict(list)

    for model, df in df_dict.items():
        for feature in features:
            values = list(df[df["Feature group"] == feature]["Shap values"])
            scaled_values = [i * scaling_factor for i in values]
            data[model].append(scaled_values)

    for idx, f in enumerate(features):
        if "Pressure" in f:
            features[idx] = f.replace("Pressure", "Sea Level Pressure")

    return data, features


with open(
    "/home/maximilianerautenstrauss/git-private/forecasting/forecasting/shap_values_df_mlp.p",
    "rb",
) as file:
    shap_obj_mlp = pickle.load(file)

with open(
    "/home/maximilianerautenstrauss/git-private/forecasting/forecasting/shap_values_df_cnn.p",
    "rb",
) as file:
    shap_obj_cnn = pickle.load(file)

with open(
    "/home/maximilianerautenstrauss/git-private/forecasting/forecasting/shap_values_df_rf.p",
    "rb",
) as file:
    shap_obj_rf = pickle.load(file)

shap_obj_df_dict = {"CNN": shap_obj_cnn, "MLP": shap_obj_mlp, "RF": shap_obj_rf}
generate_boxplots(shap_obj_df_dict, scaling_factor=100)
