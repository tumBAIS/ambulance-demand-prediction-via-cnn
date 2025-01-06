from ast import literal_eval
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from skopt import Space
from skopt.space import Categorical, Integer, Real
from tensorflow.keras import layers, losses, metrics, optimizers

FEATURES_RESULTS_LABEL = "features"


def read_space(csv_name: str) -> List[Union[Integer, Real, Categorical]]:
    """Generates space array with Space objects from csv

    Parameters:
    csv_name: str
        name of csv stating space characteristics

    Returns:
    space: list(skopt.space.Real, skopt.space.Integer or skopt.space.Categorical)
        List with space dimensions (Real, Categorical and Integer)
    """
    x = pd.read_csv(csv_name, delimiter=";")
    space = []
    for row in x.itertuples():

        if row.type == "Integer":
            space.extend(
                [
                    Integer(
                        int(row.lower_bound),
                        int(row.upper_bound),
                        transform=row.transform,
                        name=row.name,
                    )
                ]
            )
        elif row.type == "Real":
            space.extend(
                [
                    Real(
                        float(row.lower_bound),
                        float(row.upper_bound),
                        transform=row.transform,
                        name=row.name,
                    )
                ]
            )
        elif row.type == "Categorical":
            categories = row.categories.split(",")
            for i, category in enumerate(categories):
                categories[i] = convert_datatype(category)
            space.extend([Categorical(categories, name=row.name)])
    return space


def convert_datatype(n: Any) -> Optional[Union[bool, int, float, str]]:
    """Converts string input parameter into corresponding data type
    (e.g. '1.1' -> float, '1' -> int)

    Parameters:
    n: str
        Element to be converted

    Returns:
    x: Optional[Union[bool, int, float, str]]
        Data converted to corresponding data type

    """
    try:
        if n in {"true", "True"}:
            x = True
        elif n in {"false", "False"}:
            x = False
        elif n.isdigit():
            x = int(n)
        else:
            x = float(n)
    except ValueError:
        x = n
    return x


def read_x0(csv_name: str) -> List[Any]:
    """Generates list with parameter combinations read as string from <csv_name>

    Parameters:
    csv_name: str
        Name of csv with x0: "[[0,1,0,2],[1,2,1,0]]"

    Returns:
    x0: List[Any]
        List with parameter combinations
    """
    with open(csv_name) as f:
        lines = f.readlines()
    x0 = literal_eval(lines[0])
    return x0


def read_y0(csv_name: str) -> List[float]:
    """Returns inital values x0 and y0 for input in BO given a results file

    Parameters:
    csv_name: str
        csv name with results (each result in own line)

    Returns:
    y0: List[float]
        list with results
    """

    with open(csv_name) as file:
        lines = file.readlines()

    lines = str([line.rstrip() for line in lines])
    lines = literal_eval(lines)
    lines = [float(i) for i in lines]

    return lines


def replace_element(e: Any) -> str:
    """Replaces element e by 'readable' name
    (e.g. keras.metrics.RootMeanSquaredError -> RMSE)

    Any:
    e: Any
        Element to be replaced by readable name

    Returns:
    e: str
        Readable name (e.g. to 'MSE', 'RMSE', etc.)
    """

    # object dictionary (key = object that needs to be replaced, value: value replacing corresponding object)
    act_obj = {
        layers.ReLU: "relu",
        layers.ELU: "elu",
        layers.LeakyReLU: "leakyrelu",
        optimizers.Adam: "Adam",
        optimizers.SGD: "SGD",
        losses.MSE: "MSE",
        losses.MeanSquaredError: "MSE",
        metrics.RootMeanSquaredError: "RMSE",
    }

    # string dictionary (key = if this substring is contained in e, it is replaced by its value)
    act_str = {
        ".LeakyReLU": "leakyrelu",
        "leaky_re_lu": "leakyrelu",
        ".ReLU": "relu",
        "re_lu": "relu",
        ".ELU": "elu",
        "elu_": "elu",
        "SGD": "SGD",
        "Adam": "Adam",
        "RootMeanSquaredError": "RMSE",
        "root_mean_squared_error": "RMSE",
        "MeanSquaredError": "MSE",
        "mean_squared_error": "MSE",
    }

    if e is None:
        return e
    if callable(e):
        if hasattr(e, "__name__"):
            e = e.__name__
        elif hasattr(e, "name"):
            e = e.name
    elif not isinstance(e, str) and (type(e) in act_obj):
        return act_obj[type(e)]
    if isinstance(e, str):
        for key, value in act_str.items():
            if key in e:
                return value
    return e


def replace_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces all activation, metrics and optimizer outputs by
    'readable' name (e.g. layers.keras...ReLu -> relu)

    Parameters:
    df: pandas.Dataframe
        Dataframe with values to be replaced

    Returns:
    df: pandas.Dataframe
        Dataframe in which all activations and optimizer names are adapted
        (e.g. to 'elu', 'relu', etc.)
    """

    column_names = [
        s
        for s in df.columns
        if (
            ("activation" in s)
            or ("optimizer" in s)
            or ("metrics" in s)
            or ("loss_function" in s)
        )
    ]
    for col in column_names:
        for row in range(len(df)):
            element = df.at[row, col]
            if isinstance(element, list):
                for idx, _ in enumerate(element):
                    df.at[row, col][idx] = replace_element(element[idx])
            else:
                df.at[row, col] = replace_element(element)
    return df


def results_extract_x0_y0(
    space: Space,
    y0_name: str,
    results_csv: str,
    feature_selection: bool = False,
    feat: List[str] = [],
) -> Tuple[List[Any], List[Any]]:
    """Returns inital values x0 and y0 for input in BO given a results file

    Parameters:
    space: Space
        Space object to be obtimized
    y0_name: str
        Name of variable to be optimized (e.g. 'mean_squared_error').
        Must correspond to column name in results_csv
    results_csv: str
        csv name of results file from which results should be extracted
    feature_selection: Bool
        True if applied features should be included in x0 as booleans
    feat: list(str)
        List of strings with feature names of applied features (must correspond to names in results file).
        Can be extracted from filtered feature_overview of Data_Storage

    Returns:
    x0: List[Any]
        List with x values (hyperparameter settings)
    y0: List[float]
        List with y values (function values)
    """
    # get names of variables to be optimized by applying BO
    space_names = space.dimension_names

    # read results
    results = pd.read_csv(results_csv, converters={FEATURES_RESULTS_LABEL: pd.eval})

    # if included features should be added as boolean array,
    # generate a column for each feature and its boolean value
    if feature_selection:
        if len(feat) == 0:
            raise ValueError("Please provide features to be extracted.")
        results["features_bool"] = results.apply(
            lambda row: [int(f in row[FEATURES_RESULTS_LABEL]) for f in feat], axis=1
        )
        bool_feat_df = pd.DataFrame(results["features_bool"].to_list(), columns=feat)
        results = pd.concat([results, bool_feat_df], axis=1)
        space_names.extend(feat)

    results = replace_objects(results)

    # filter results such that only relevant variables are contained
    x0 = results[space_names]
    y0 = results[y0_name]

    return x0.values.tolist(), y0.values.tolist()
