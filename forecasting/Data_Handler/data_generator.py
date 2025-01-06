# pylint: disable=redefined-variable-type
import calendar
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Data_Handler.event_handler import EventCube
from Data_Handler.feature_names import (
    FEATURE_SETTINGS_LABEL_CATEGORY, FEATURE_SETTINGS_LABEL_FEATURE_W_TIME,
    FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME,
    FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME,
    FEATURE_SETTINGS_LABEL_INCLUSION, FEATURE_SETTINGS_LABEL_INPUT_SHAPE,
    FEATURE_SETTINGS_LABEL_ONE_HOT_ENCODING, FEATURE_SETTINGS_LABEL_TIME,
    FEATURE_SETTINGS_LABEL_TRANSPOSE, Feature_w_Time, Feature_wo_Time,
    FeatureCategory, TimeCategory, assemble_feature_w_time)
from Data_Handler.heatmap_handler import HeatmapCube
from Data_Handler.holiday_handler import HolidayHandler
from Data_Handler.weather_handler import WeatherHandler
from Models.types import DataType


def read_features_df(csv_name: str) -> pd.DataFrame:
    """
    Reads features from csv-file and returns dataframe with
    all feature information of features to be considered

    Parameters:
    csv_name: str
        name of csv-file containing feature information,
        must include column 'Include', 'One_Hot_Encoding' and 'Transpose'

    Returns:
    features_df: pd.DataFrame
        Dataframe with all feature information of features to be considered
    """

    def replace_feature_wo_time(x: str) -> Feature_wo_Time:
        return Feature_wo_Time[x.upper()]

    def replace_feature_category(x: str) -> FeatureCategory:
        return FeatureCategory[x.upper()]

    def replace_time_category(x: str) -> TimeCategory:
        return TimeCategory[x.upper()]

    # Read Input Features
    features_df = pd.read_csv(
        csv_name,
        sep=";",
        dtype={
            FEATURE_SETTINGS_LABEL_ONE_HOT_ENCODING: np.bool,
            FEATURE_SETTINGS_LABEL_TRANSPOSE: np.bool,
            FEATURE_SETTINGS_LABEL_INCLUSION: np.bool,
        },
    )

    try:
        # replace str values by objects
        features_df[FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME] = features_df[
            FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME
        ].apply(replace_feature_wo_time)
        features_df[FEATURE_SETTINGS_LABEL_CATEGORY] = features_df[
            FEATURE_SETTINGS_LABEL_CATEGORY
        ].apply(replace_feature_category)
        features_df[FEATURE_SETTINGS_LABEL_TIME] = features_df[
            FEATURE_SETTINGS_LABEL_TIME
        ].apply(replace_time_category)
        features_w_time = [
            assemble_feature_w_time(
                features_df.iloc[idx][FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME],
                features_df.iloc[idx][FEATURE_SETTINGS_LABEL_TIME],
            )
            for idx, _ in features_df.iterrows()
        ]
        features_wo_time_name = [
            features_df.iloc[idx][FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME].name
            for idx, _ in features_df.iterrows()
        ]
    except KeyError as exc:
        raise KeyError(f"Check input data in {csv_name}") from exc

    features_df[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME] = features_w_time
    features_df[FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME] = features_wo_time_name
    return features_df


def add_feature_data_to_data_storage(
    features_csv: str,
    data_storage: Any,
    data_types: Optional[List[DataType]],
    total_look_back: int,
    time_intervals: List[pd.Interval],
    time_intervals_medic: List[pd.Interval],
    latitude_intervals: pd.interval_range,
    longitude_intervals: pd.interval_range,
    nrRows: int,
    nrCols: int,
    calldata: str,
    filter_dict: Dict[str, List[str]],
    shorten_one_hot: bool = True,
):
    """
    Generates data given in feature settings (read from <features_csv>)
    and stores it in <data_storage>.
    Sets feature_df in <data_storage> and updates data (X) based on feature settings.

    Parameters:
    features_csv: str
        Path containing feature settings
    data_storage: ds.DataStorage
        DataStorage instance in which data should be saved
    data_types: Optional[List[DataType]]
        DataTypes for which data should be genereated and saved
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time periods that are examined
    time_intervals_medic: List[pd.Interval]
        list of time periods that can be used for medic method
    nrRows: int
        number of horizontal area divisions
    nrCols: int
        number of vertical area divisions
    shorten_one_hot: bool
        If false, vector of length 24/7/12 is returned for hour/day/month (default is True)
        If true, vector length is adapted to the number of distinct time values
    latitude_intervals: pd.interval_range
        latitude values for vertical area divisions
    longitude_intervals: pd.interval_range
        longitude values for horizontal area divisions
    calldata: str
        name of dataset, supported: 'Seattle'
    filter_dict: Dict[str, List[str]]
        dict(key: str (column name in call data), values: List of column values that should be filtered
        dictionary including column names and values (in calldata) to filter
        e.g. {'Type': ['Medic Response']})
    """

    features_df = read_features_df(features_csv)
    features = [Feature_wo_Time.AMBULANCE_DEMAND] + list(
        set(features_df[FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME])
    )

    # data basis
    heatmap_cube = HeatmapCube(calldata).get_heatmap_cube(
        filter_dict, time_intervals, latitude_intervals, longitude_intervals
    )
    event_cube = EventCube(
        time_intervals, latitude_intervals, longitude_intervals, nrRows, nrCols
    ).get_event_cube()
    holiday_data = HolidayHandler(time_intervals).get_holiday_data()
    weather_data = WeatherHandler(time_intervals).get_weather_data()

    if data_types is None:
        data_types = list(DataType)

    # Individual features require different functions to extract relevant data
    for feature in features:

        if feature == Feature_wo_Time.AMBULANCE_DEMAND:
            feature_w_time = assemble_feature_w_time(feature, TimeCategory.HISTORY)
            for data_type in data_types:
                X, y, value_set = generate_Xy_calls(
                    total_look_back, time_intervals, heatmap_cube, data_type
                )
                max_calls = np.max(X)
                data_storage.X_dict[data_type][feature_w_time] = (
                    X / max_calls
                )  # calls scaled between [0,1]
                data_storage.y[data_type] = y
                data_storage.X_value_sets[data_type][feature_w_time] = value_set

        elif feature in {
            Feature_wo_Time.HOURS,
            Feature_wo_Time.WEEKDAYS,
            Feature_wo_Time.MONTHS,
        }:
            feature_w_time = assemble_feature_w_time(feature, TimeCategory.PREDICTION)

            for data_type in data_types:

                X, value_set = generate_X_time(
                    feature,
                    total_look_back,
                    time_intervals,
                    nrRows,
                    nrCols,
                    data_type,
                    shorten_one_hot,
                )
                data_storage.X_dict[data_type][feature_w_time] = X
                data_storage.X_value_sets[data_type][feature_w_time] = value_set

            # should be same for all data_types
            features_df.loc[
                features_df[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME] == feature_w_time,
                FEATURE_SETTINGS_LABEL_INPUT_SHAPE,
            ] = len(data_storage.X_value_sets[data_type][feature_w_time])

        elif feature in {
            Feature_wo_Time.PUBLIC_HOLIDAYS,
            Feature_wo_Time.SCHOOL_HOLIDAYS,
        }:

            feature_w_time = assemble_feature_w_time(feature, TimeCategory.PREDICTION)

            for data_type in data_types:
                X = generate_X_pred(
                    total_look_back,
                    time_intervals,
                    nrRows,
                    nrCols,
                    holiday_data,
                    feature.name,
                    data_type,
                )
                data_storage.X_dict[data_type][feature_w_time] = X
                data_storage.X_value_sets[data_type][feature_w_time] = [
                    feature_w_time.name
                ]

        elif feature in {
            Feature_wo_Time.MAX_TEMP_F,
            Feature_wo_Time.AVG_TEMP_F,
            Feature_wo_Time.MIN_TEMP_F,
            Feature_wo_Time.MAX_WIND_SPEED_MPH,
            Feature_wo_Time.AVG_WIND_SPEED_MPH,
            Feature_wo_Time.MIN_WIND_SPEED_MPH,
            Feature_wo_Time.MAX_HUMIDITY_PERCENT,
            Feature_wo_Time.AVG_HUMIDITY_PERCENT,
            Feature_wo_Time.MIN_HUMIDITY_PERCENT,
            Feature_wo_Time.MAX_DEW_POINT_F,
            Feature_wo_Time.AVG_DEW_POINT_F,
            Feature_wo_Time.MIN_DEW_POINT_F,
            Feature_wo_Time.MAX_PRESSURE_HG,
            Feature_wo_Time.AVG_PRESSURE_HG,
            Feature_wo_Time.MIN_PRESSURE_HG,
            Feature_wo_Time.TOTAL_PRECIPITATION_INTENSITY,
        }:
            feature_w_time = assemble_feature_w_time(feature, TimeCategory.PREDICTION)

            for data_type in data_types:
                X = generate_X_pred(
                    total_look_back,
                    time_intervals,
                    nrRows,
                    nrCols,
                    weather_data,
                    feature.name,
                    data_type,
                )
                data_storage.X_dict[data_type][feature_w_time] = X
                data_storage.X_value_sets[data_type][feature_w_time] = [
                    feature_w_time.name
                ]

            feature_w_time = assemble_feature_w_time(feature, TimeCategory.HISTORY)

            for data_type in data_types:
                X = generate_X_hist(
                    total_look_back,
                    time_intervals,
                    nrRows,
                    nrCols,
                    weather_data,
                    feature.name,
                    data_type,
                )
                data_storage.X_dict[data_type][feature_w_time] = X
                data_storage.X_value_sets[data_type][feature_w_time] = [
                    f"{feature_w_time.name}_t-{i}"
                    for i in range(total_look_back, 0, -1)
                ]

        elif feature == Feature_wo_Time.EVENTS:

            feature_w_time = assemble_feature_w_time(feature, TimeCategory.PREDICTION)

            for data_type in data_types:
                X_events_pred, value_set = generate_X_events(
                    total_look_back, time_intervals, event_cube, data_type
                )
                data_storage.X_dict[data_type][feature_w_time] = X_events_pred
                data_storage.X_value_sets[data_type][feature_w_time] = value_set

    # add time series data for medic method
    heatmap_cube_medic = HeatmapCube(calldata).get_heatmap_cube(
        filter_dict,
        time_intervals_medic,
        latitude_intervals,
        longitude_intervals,
    )
    time_series_per_subregion = generate_X_calls_time_series(heatmap_cube_medic)
    data_storage.time_series_medic = time_series_per_subregion

    # update_data_set = True so that method inits also X and features
    data_storage.set_features_df(features_df, update_data=True, data_types=data_types)


def generate_X_calls_time_series(
    heatmap_cube: np.ndarray,
) -> np.ndarray:
    time_series_per_subregion = np.empty(
        shape=(heatmap_cube.shape[0], heatmap_cube.shape[1]), dtype="object"
    )
    for r in range(heatmap_cube.shape[0]):
        for c in range(heatmap_cube.shape[1]):
            time_series_per_subregion[r][c] = heatmap_cube[r, c, :]
    return time_series_per_subregion


def generate_Xy_calls(
    total_look_back: int,
    time_intervals: List[pd.Interval],
    heatmap_cube: np.ndarray,
    data_type: DataType,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Returns data set with predicted data for defined feature (max temp, min temp, public holiday, etc.)

    Parameters:
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    heatmap_cube: heatmap_cube: np.ndarray
        3D array with historic ambulance demand
    data_type: DataType
        data type for which training and test data should be created

    Returns:
    X: np.ndarray
        Training set inputs
    y: np.ndarray
        Training set outputs

    """

    X = []
    y = []

    if data_type == DataType.LAYER_BASED:
        for t in range(total_look_back, len(time_intervals) - 1):
            X.append(heatmap_cube[:, :, (t - total_look_back) : t])
            y.append(heatmap_cube[:, :, t + 1])
        value_set = [
            Feature_w_Time.AMBULANCE_DEMAND_HISTORY.name + f"_({r},{c})_t-{t}"
            for r in range(heatmap_cube.shape[0])
            for c in range(heatmap_cube.shape[1])
            for t in range(total_look_back, 0, -1)
        ]

    elif data_type == DataType.INSTANCE_BASED:
        for t in range(total_look_back, len(time_intervals) - 1):
            for r in range(heatmap_cube.shape[0]):
                for c in range(heatmap_cube.shape[1]):
                    X.append(heatmap_cube[r, c, (t - total_look_back) : t])
                    y.append(heatmap_cube[r, c, t + 1])
        value_set = [
            Feature_w_Time.AMBULANCE_DEMAND_HISTORY.name + f"_t-{t}"
            for t in range(total_look_back, 0, -1)
        ]

    else:
        raise NotImplementedError(f"Datatype {data_type} not implemented.")

    # Transform and reshape historic data
    X = np.array(X)
    y = np.array(y)

    if data_type == DataType.LAYER_BASED:
        X = X.reshape(
            (len(X), heatmap_cube.shape[0], heatmap_cube.shape[1], total_look_back, 1)
        )

    return X, y, value_set


def generate_X_time(
    feature: Feature_wo_Time,
    total_look_back: int,
    time_intervals: List[pd.Interval],
    nrRows: int,
    nrCols: int,
    data_type: DataType,
    shorten_one_hot: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns data set with the corresponding time information (hour/weekday/month) of the predicted period.

    The time information always refers to the start of the interval.
    The time information is provided one hot encoded if <shorten_one_hot> is True.

    Parameters:
    feature: Feature_wo_Time
        Feature information to be extracted.
        The following time periods are supported: Hour, Weekday, Month.
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time periods that are examined
    nrRows: int
        number of horizontal area divisions
    nrCols: int
        number of vertical area divisions
    data_type: DataType
        data type for which data should be created
        (e.g. data structure for CNN differs from structure required by MLP)
    shorten_one_hot: bool, default: True
        If false, vector of length 24/7/12 is returned for hour/day/month.
        If true, vector length is adapted to the number of distinct time values

    Returns:
    X: np.ndarray
        List of one hot encoded 1D vectors including with time information
    value_set: List[str]
        List of values that have been considered (e.g. [Hour_0,Hour_8,Hour_16] for 8h time interval)
        -> NOT [Hour_0,Hour_1,Hour_2,...,Hour_24])
    """

    X_int = []

    # X_int2 used for storing data
    if data_type == DataType.INSTANCE_BASED:
        X_int2 = []

    for t in range(total_look_back, len(time_intervals) - 1):

        if feature == Feature_wo_Time.HOURS:
            X_int.append((time_intervals[t + 1].left.hour))  # Values 0-23
        elif feature == Feature_wo_Time.WEEKDAYS:
            X_int.append((time_intervals[t + 1].left.day_of_week))  # Values 0-6
        elif feature == Feature_wo_Time.MONTHS:
            X_int.append((time_intervals[t + 1].left.month) - 1)  # Values 1-12
        else:
            raise ValueError("Feature: ", feature, " unknown")

        if data_type == DataType.INSTANCE_BASED:
            X_int2.extend([X_int[-1]] * (nrRows * nrCols))

    if data_type == DataType.INSTANCE_BASED:
        X_int = X_int2

    if shorten_one_hot:
        # length of one-hot encoded vector corresponds to the number of distinct values
        # e.g. if we have 8h intervals, we only have intervals such as 0-8, 8-16, 16-0
        # in this example, a vector of length 3 is created
        # save all values that have been *observed* (e.g. [0,8,16] for a 8h time interval)
        value_set = list(sorted(set(X_int)))
        length_one_hot = len(value_set)
        for h, _ in enumerate(X_int):
            X_int[h] = value_set.index(X_int[h])

    else:

        # save all values *considered* (e.g. [0,1...,23], even for a 8h time interval)
        if feature == Feature_wo_Time.HOURS:
            length_one_hot = 24
        elif feature == Feature_wo_Time.WEEKDAYS:
            length_one_hot = 7
        elif feature == Feature_wo_Time.MONTHS:
            length_one_hot = 12
        else:
            raise ValueError("Feature: ", feature, " unknown")

        value_set = list(range(length_one_hot))

    if feature == Feature_wo_Time.HOURS:
        value_set = [("Hour_" + str(i)) for i in value_set]
    elif feature == Feature_wo_Time.WEEKDAYS:
        value_set = [calendar.day_name[i] for i in value_set]
    elif feature == Feature_wo_Time.MONTHS:
        value_set = [calendar.month_name[i + 1] for i in value_set]
    else:
        raise ValueError("Feature: ", feature, " unknown")

    # Onehot Encoding & Reshape Weekdays, Months
    X = np.zeros(shape=(len(X_int), length_one_hot))
    X[np.arange(len(X_int)), X_int] = 1
    X = X.reshape(
        (
            len(X),
            length_one_hot,
        )
    )

    return X, value_set


def generate_X_events(
    total_look_back: int,
    time_intervals: List[pd.Interval],
    event_cube: np.ndarray,
    data_type: DataType,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns data set with event prediction

    Parameters:
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time periods that are examined
    event_cube: 3D-array
        3D array with event data (spatial and temporal)
    data_type: types.DataType
        data type for which data should be created
        (e.g. data structure for CNN differs from structure required by MLP)

    Returns:
    X_pred: np.ndarray
        Array of 2D arrays with event predictions
    value_set: List[str]
        List of values that have been considered (e.g. [Hour_0,Hour_8,Hour_16] for 8h time interval)
        -> NOT [Hour_0,Hour_1,Hour_2,...,Hour_24])

    :raises NotImplementedError: data_type is a unknown DataType.
    """
    X_pred = []
    if data_type == DataType.LAYER_BASED:
        for t in range(total_look_back, len(time_intervals) - 1):
            X_pred.append(event_cube[:, :, (t + 1)])
        value_set = [
            Feature_w_Time.EVENTS_PREDICTION.name + f"_({r},{c})"
            for r in range(event_cube.shape[0])
            for c in range(event_cube.shape[1])
        ]

    elif data_type == DataType.INSTANCE_BASED:
        for t in range(total_look_back, len(time_intervals) - 1):
            for r in range(event_cube.shape[0]):
                for c in range(event_cube.shape[1]):
                    X_pred.append(event_cube[r, c, (t + 1)])
        value_set = [Feature_w_Time.EVENTS_PREDICTION.name]

    else:
        raise NotImplementedError(f"Datatype {data_type} not implemented.")

    X_pred = np.array(X_pred)
    if data_type == DataType.LAYER_BASED:
        X_pred = X_pred.reshape(
            (X_pred.shape[0], event_cube.shape[0], event_cube.shape[1], 1)
        )

    return X_pred, value_set


def generate_Xy_hist_cube(
    total_look_back: int, time_intervals: List[pd.Interval], heatmap_cube: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns data set with call volumes (with defined lookback) (3D)
    and corresponding call colume in next period (2D)

    Parameters:
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    heatmap_cube: np.ndarray
        3D array with historic ambulance demand

    Returns:
    X_hist: List[np.ndarray]
        Training set inputs with historic call volume (list of 3D arrays)
    y: List[np.ndarray]
        Training set outputs (predictions) (list of 2D arrays)
    """

    # arrays for historical data
    X_hist = []
    y = []

    for t in range(total_look_back, len(time_intervals) - 1):

        # historic data
        X_hist.append(heatmap_cube[:, :, (t - total_look_back) : t])
        y.append(heatmap_cube[:, :, t + 1])

    X_hist = np.array(X_hist)
    y = np.array(y)

    # Reshape historic data
    X_hist = X_hist.reshape(
        shape=(
            len(X_hist),
            heatmap_cube.shape[0],
            heatmap_cube.shape[1],
            total_look_back,
            1,
        )
    )

    return X_hist, y


def generate_X_hist(
    total_look_back: int,
    time_intervals: List[pd.Interval],
    nrRows: int,
    nrCols: int,
    data: pd.DataFrame,
    feature: str,
    data_type: DataType,
) -> np.ndarray:
    """
    Returns data set with historic data (with defined lookback)
    for defined feature (max temp, min temp, etc.)

    Parameters:
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: list(pd.Interval)
        list of time intervals that are included
    nrRows: int
        number of horizontal area divisions
    nrCols: int
        number of vertical area divisions
    data: pd.DataFrame
        Dataframe including data, feature_name must be included as column name
    data_type: DataType
        data type for which data should be created
        (e.g. data structure for CNN differs from structure required by MLP)
    feature: str
        Column name of feature for which data is wanted

    Returns:
    X_hist: np.ndarray
        Training set inputs with corresponding data (column: feature)
    """

    # arrays for historical data
    X_hist = []

    for t in range(total_look_back, len(time_intervals) - 1):

        # historic data
        if data_type == DataType.LAYER_BASED:
            X_hist.append(data[feature][(t - total_look_back) : t])

        elif data_type == DataType.INSTANCE_BASED:
            X_hist.extend(
                [data[feature][(t - total_look_back) : t]] * (nrRows * nrCols)
            )

        else:
            raise NotImplementedError(f"Datatype {data_type} not implemented.")

    X_hist = np.array(X_hist)

    X_hist = X_hist.reshape((len(X_hist), 1, 1, total_look_back, 1))

    return X_hist


def generate_X_pred(
    total_look_back: int,
    time_intervals: List[pd.Interval],
    nrRows: int,
    nrCols: int,
    data: pd.DataFrame,
    feature: str,
    data_type: DataType,
) -> np.ndarray:
    """
    Returns data set with predicted data for defined feature (max temp, min temp, public holiday, etc.)

    Parameters:
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    nrRows: int
        number of horizontal area divisions
    nrCols: int
        number of vertical area divisions
    data: pd.DataFrame
        Dataframe including data, feature_name must be included as column name
    feature: str
        Column name of feature for which data is wanted
    data_type: DataType
        data type for which data should be created (e.g. data structure for CNN differs from structure required by MLP)

    Returns:
    X_pred: np.ndarray
        Training set inputs with predicted data

    """
    X_pred = []
    for t in range(total_look_back, len(time_intervals) - 1):

        if data_type == DataType.LAYER_BASED:
            X_pred.append(data[feature][t + 1])

        elif data_type == DataType.INSTANCE_BASED:
            X_pred.extend([data[feature][t + 1]] * (nrRows * nrCols))

        else:
            raise NotImplementedError(f"Datatype {data_type} not implemented.")

    X_pred = np.array(X_pred)
    return X_pred
