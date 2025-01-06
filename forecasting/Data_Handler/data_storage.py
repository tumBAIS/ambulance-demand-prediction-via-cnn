import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Data_Handler.data_generator import add_feature_data_to_data_storage
from Data_Handler.feature_names import (
    FEATURE_SETTINGS_LABEL_CONCATENATION,
    FEATURE_SETTINGS_LABEL_FEATURE_W_TIME,
    FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME,
    FEATURE_SETTINGS_LABEL_INCLUSION, Feature_w_Time)
from Models.types import DataType
from sklearn.model_selection import train_test_split


class DataStorage:
    """
    Storage for data used by (different) model(s) incl. X, y

    Parameters:
        X_dict: Dict[DataType, Dict[Feature_w_Time, np.ndarray]]
            Training set inputs for features and historic call volume (e.g. {'Hours_Prediction': [[0,0,1],[1,0,0]...]})
            Contains ALL possible inputs (even if not included in actual run)
        X_value_sets: Dict[DataType, Dict[Feature_w_Time, List[str]]]
            Considered values for features (e.g. {'Hours_Prediction': [0,8,16]} for 8h time intervals)
            Contains ALL possible inputs (even if not included in actual run)
        X: np.ndarray, default: None
            Contains input data (values) for ML algorithms
            Contains only inputs marked as 'Included' in features_df and filtered by filter_array, if provided
            Set/Overwritten if update_data_set is True and features_df not None
        y: np.ndarray, default: None
            Contains output data (values) for ML algorithms
            Set/Overwritten if update_data_set is True and features_df not None
        filter_array: np.ndarray
            Boolean array containing which features should be included
            Length of array should correspond to number ob features marked as "Included" in features_df
        features_df: pd.DataFrame, default: None
            Dataframe with all feature information of features to be considered
        update_data_set: bool, default: True
            If True and features_df is provided, X is overwritten (also based on filter_array if provided)
        time_series_medic: np.ndarray
            Numpy array including time series for each subregion with shape nrRows x nrCols x nr. of time periods considered for medic method
    """

    def __init__(
        self,
        X_dict: Dict[DataType, Dict[Feature_w_Time, np.ndarray]] = defaultdict(dict),
        X_value_sets: Dict[DataType, Dict[Feature_w_Time, List[str]]] = defaultdict(
            dict
        ),
        y: Dict[DataType, np.ndarray] = {},
        X: Dict[DataType, np.ndarray] = {},
        filter_array: Optional[np.ndarray] = None,
        features_df: Optional[pd.DataFrame] = None,
        update_data: bool = True,
        time_series_medic: Optional[np.ndarray] = None,
    ):

        self.X_dict = X_dict
        self.X_value_sets = X_value_sets
        self.filter_array = filter_array
        self.X = X
        self.y = y
        self.update_data = update_data
        self.time_series_medic = time_series_medic
        self.set_features_df(features_df, update_data)

    def get_X(self, data_type: DataType) -> np.ndarray:
        return self.X[data_type]

    def get_y(self, data_type: DataType) -> np.ndarray:
        return self.y[data_type]

    def get_time_series_medic(self):
        return self.time_series_medic

    def get_features_overview(
        self, include_historic_ambulance_demand: bool = False
    ) -> List[Feature_w_Time]:
        """Returns list of features

        Parameters:
        include_historic_ambulance_demand: bool, default: False
            If True, Feature_w_Time.AMBULANCE_DEMAND_HISTORY and corresponding
            feature names are included in dict

        Returns:
        features_overview: List[Feature_w_Time]
            List containing all Features
        """

        features_df = self.get_filtered_features_df()
        self.sort_features_df(features_df)

        if include_historic_ambulance_demand:
            return [Feature_w_Time.AMBULANCE_DEMAND_HISTORY] + list(
                features_df[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME]
            )
        return list(features_df[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME])

    def get_features_dict(
        self, data_type: DataType, include_historic_ambulance_demand: bool = False
    ) -> Dict[Feature_w_Time, List[str]]:
        """Returns dictionary containing filtered and sorted features and its corresponding values

        Parameters:
        data_type: DataType
            DataType for which features_dict is wanted (feature names differ for DataTypes)
        include_historic_ambulance_demand: bool, default: False
            If True, Feature_w_Time.AMBULANCE_DEMAND_HISTORY and corresponding
            feature names are included in dict

        Returns:
        features_dict: Dict[Feature_w_Time, List[str]]
            Dictionary containing Feature_w_Time instances as keys and corresponding
            features names as values (e.g. ["HOUR_0", "HOUR_8", "HOUR_16"])
        """

        features_df = self.get_filtered_features_df()
        self.sort_features_df(features_df)

        features_dict = {}
        if include_historic_ambulance_demand:
            feature_groups = [Feature_w_Time.AMBULANCE_DEMAND_HISTORY] + list(
                features_df[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME]
            )
        else:
            feature_groups = list(features_df[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME])

        for feature in feature_groups:
            features_dict[feature] = self.X_value_sets[data_type][feature]

        return features_dict

    def set_features_df(
        self,
        features_df: pd.DataFrame,
        update_data: bool = True,
        data_types: Optional[List[DataType]] = None,
    ) -> None:
        """Setter for <self.feature_df>, simultaneously updates data (X)
        for given models based on <features_df> if <update_data> = True

        Parameters:
        features_df: pd.DataFrame
            Dataframe with all feature information of features to be considered
        update_data: bool, default: True
            If True, X is overwritten based on features_df
        set_features_df: Optional[List[DataType]], default: None
            List of data types for which input data (X,y) must be updated based on features_df
            If None, all data types are updated
        """
        self.features_df = features_df

        if self.features_df is not None and update_data:
            # make sure features_df is in right order (needed for CNN)
            self.sort_features_df(self.features_df)
            self.update_data_set(sortdf=False, data_types_to_update=data_types)

    def sort_features_df(self, features_df: pd.DataFrame):
        """Sorts features_df according to concatenation type and feature name

        Parameters:
            features_df: pd.DataFrame
                Dataframe to be sorted inplace
        """
        features_df.sort_values(
            by=[
                FEATURE_SETTINGS_LABEL_CONCATENATION,
                FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME,
            ],
            ascending=[False, True],
            inplace=True,
        )

    def get_filtered_features_df(self) -> pd.DataFrame:
        """Creates a copy of <self.features_df> and filters it according to column
        <FEATURE_SETTINGS_LABEL_INCLUSION> and <self.filter_array>

        If feature is marked as included (1 or True), feature will remain in dataframe,
        otherwise this feature will be removed.
        Dataframe is further filtered according to filter_array
        (Boolean array containing which features should be included.
        Length of filter_array should correspond to number ob features marked as "Included" in self.features_df).

        Returns:
        features_df2: pd.DataFrame (if onlynames = False) or pd.Series (if onlynames = True)
            Copy of dataframe filtered by <FEATURE_SETTINGS_LABEL_INCLUSION> column
            and filter_array
        """
        features_df2 = self.features_df[
            self.features_df[FEATURE_SETTINGS_LABEL_INCLUSION]
        ]

        if self.filter_array is not None:
            if len(self.filter_array) != len(features_df2):
                raise ValueError(
                    "Check filter array and features_df filtering (inclusion column)"
                )
            features_df2 = features_df2[self.filter_array]

        return features_df2

    def get_data_types_in_X_dict(self) -> List[DataType]:
        """Returns list of data types saved in <X_dict> in self (DataStorage)

        Returns:
        data_types: List[DataType]
            List of data types saved in <X_dict> in self (DataStorage)
        """
        data_types = list(self.X_dict.keys())
        return data_types

    def features_dict_to_filter_array(
        self, features_dict: Dict[Feature_w_Time, List[str]]
    ) -> np.ndarray:
        """Transform <features_dict> into <filter_array>
        Does not change self.features_df but saves applied <filter_array> in self

        Parameters:
        features_dict: Dict[Feature_w_Time, List[str]]
            Dictionary containing Feature_w_Time instances as keys and corresponding
            features names as values (e.g. ["HOUR_0", "HOUR_8", "HOUR_16"])

        Returns:
        filter_array: Optional[np.ndarray], default: None
            Array stating which features should be included when updating X
            len(filter_array) must equal len(self.features_df[FEATURE_SETTINGS_LABEL_INCLUSION])
        """
        # Filter only included features
        features_df2 = self.features_df[
            self.features_df[FEATURE_SETTINGS_LABEL_INCLUSION]
        ]

        # Add 1 if feature should be included (=given in features_dict), 0 otherwise
        filter_array = []
        for _, row in features_df2.iterrows():
            if row[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME] in features_dict.keys():
                filter_array.append(1)
            else:
                filter_array.append(0)

        return np.array(filter_array)

    def update_data_set(
        self,
        filter_array: Optional[np.ndarray] = None,
        sortdf: bool = True,
        data_types_to_update: Optional[List[DataType]] = None,
    ) -> None:
        """Updates X for <data_types_to_update> based on <filter_array>
        Does not change self.features_df but saves applied <filter_array> in self

        Parameters:
        filter_array: Optional[np.ndarray], default: None
            Array stating which features should be included when updating X
            len(filter_array) must equal len(self.features_df[FEATURE_SETTINGS_LABEL_INCLUSION])
        sortdf: bool, default: True
            If true, features_df is sorted (based on order required by X)
        data_types_to_update: Optional[List[DataType]], default: None
            If None, all DataTypes are updated. Can be limited by passing only specific DataTypes as list

        """

        # own dataset X for different models
        if not data_types_to_update:
            data_types = list(DataType)
        elif not isinstance(data_types_to_update, list):
            data_types = [data_types_to_update]
        else:
            data_types = data_types_to_update

        self.X = defaultdict(list)

        # Filter only included features
        features_df2 = self.features_df[
            self.features_df[FEATURE_SETTINGS_LABEL_INCLUSION]
        ]

        # Filter dataframe using filter array (needed for BO)
        if filter_array is not None:
            if len(filter_array) == len(features_df2):
                filter_array_transformed = np.array(filter_array, dtype=bool)
                features_df2 = features_df2[filter_array_transformed]
                self.filter_array = filter_array_transformed
            else:
                raise ValueError(
                    f"Filter array length {len(filter_array)} != df length {len(features_df2)}"
                )

        # for CNN, input data must be added in certain order.
        # for this reason, we sort the dataframe correspondingly
        # if function is called when setting features_df, df is already sorted (therefore sortdf=False)
        if sortdf:
            self.sort_features_df(features_df2)

        # for data generation we need to add historic ambulance demand (normally not a feature)
        features = [Feature_w_Time.AMBULANCE_DEMAND_HISTORY]
        features.extend(list(features_df2[FEATURE_SETTINGS_LABEL_FEATURE_W_TIME]))

        for data_type in data_types:

            for feature in features:
                self.X[data_type].append(self.X_dict[data_type][feature])

            if data_type == DataType.INSTANCE_BASED:

                l = [self.X_dict[data_type][k] for k in features]
                e = list(zip(*l))
                self.X[data_type] = np.array(
                    [np.concatenate(np.array(x, dtype=object), axis=None) for x in e]
                )


def generate_training_test_data(
    *X,
    y: np.ndarray,
    test_size: float,
    random_state: int,
    data_type: DataType = DataType.LAYER_BASED,
    shuffle_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divides given inputs into training and test set
    (applying sklearn.model_selection.train_test_split)

    Parameters:
    *X: lists
        Input features e.g. X_hist, X_weekdays, etc.
    y: np.ndarray
        Targets
    test_size: float
        Pecentage share of data that should be used for testing
    random_state: int
        Random number defining how to split data set
    data_type: types.DataType, default: DataType.LAYER_BASED
        Data type for which training and test data should be created
    shuffle_data: bool default: False
        If False, data is not shuffled before dividing data into training/test set

    Returns:
    X_train: np.ndarray
        Training set inputs
    y_train: np.ndarray
        Training set outputs
    X_test: np.ndarray
        Test set inputs
    y_test: np.ndarray
        Test set outputs
    """

    if data_type == DataType.LAYER_BASED:
        inputs = [*X, y]

        (*outputs, y_train, y_test) = train_test_split(
            *inputs,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle_data,
        )

        X_train = []
        X_test = []

        for o in range(0, len(outputs), 2):
            X_train.append(outputs[o])
            X_test.append(outputs[o + 1])

    elif data_type == DataType.INSTANCE_BASED:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle_data
        )

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

    else:
        raise NotImplementedError(f"Datatype {data_type} not implemented.")

    return X_train, y_train, X_test, y_test


# pylint: disable=too-many-arguments
def __init_data_storage(
    total_look_back: int,
    time_intervals: List[pd.Interval],
    time_intervals_medic: List[pd.Interval],
    latitude_intervals: pd.interval_range,
    longitude_intervals: pd.interval_range,
    nrRows: int,
    nrCols: int,
    calldata: str,
    filter_dict: Dict[str, List[str]],
    features_csv: str,
    shorten_one_hot: bool = True,
    data_types: Optional[List[DataType]] = None,
) -> DataStorage:
    """Generates data basis from different sources (event cube, call cube etc.)
    and stores data in DataStorage instance, if wanted.

    Only features are included, that are marked as "Included"
    Dimensions of 1D data given in <features_csv> are overwritten if <shorten_one_hot> = True
    and not "all" values are used.
    (e.g. for time intervals of 8h, only 3 values are needed (e.g. 0-8,8-16,16-0), not 24)

    Parameters:
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    time_intervals_medic: List[pd.Interval]
        list of time periods that can be used for medic method
    latitude_intervals: pd.interval_range
        latitude values for vertical area divisions
    longitude_intervals: pd.interval_range
        longitude values for horizontal area divisions
    nrRows: int
        number of horizontal area divisions (=len(longitude_intervals))
    nrCols: int
        number of vertical area divisions (=len(latitude_intervals))
    calldata: str
        name of dataset, supported: 'Seattle'
    filter_dict: Dict[str, List[str]]
        dict(key: str (column name in call data), values: List of column values that should be filtered
        dictionary including column names and values (in calldata) to filter
        e.g. {'Type': ['Medic Response']})
    features_csv: str
        name of csv file in which feature information is saved (is later read into features dataframe)
        Must include columns:
            - Feature
            - Time
            - Input_Shape
            - One_Hot_Encoding
            - Concatenation
            - Transpose
            - Include
            - Category
    shorten_one_hot: bool, optional, default: True
        If false, vector of length 24/7/12 is returned for hour/day/month, correspondingly
        If true, vector length is adapted to the number of distinct time values
    data_types: List[types.DataType], default: None
        data type for which data should be created
        (e.g. data structure for CNN differs from structure required by MLP)

    Returns:
    data_storage: DataStorage
        DataStorage instance with saved data for provided models
    """
    # features
    data_storage = DataStorage()

    add_feature_data_to_data_storage(
        features_csv,
        data_storage,
        data_types,
        total_look_back,
        time_intervals,
        time_intervals_medic,
        latitude_intervals,
        longitude_intervals,
        nrRows,
        nrCols,
        calldata,
        filter_dict,
        shorten_one_hot,
    )

    return data_storage


def get_data_storage(
    load_data_storage_path: str,
    save_data_storage_path: str,
    total_look_back: int,
    time_intervals: List[pd.Interval],
    time_intervals_medic: List[pd.Interval],
    latitude_intervals: pd.interval_range,
    longitude_intervals: pd.interval_range,
    nrRows: int,
    nrCols: int,
    calldata: str,
    filter_dict: Dict[str, List[str]],
    features_csv: str,
    shorten_one_hot: bool,
    data_types: List[DataType],
) -> DataStorage:
    """Loads DataStorage instance, or creates (and saves) a new instance

    Parameters:
    load_data_storage_path: str, default: None
        Path to DataStorage instance (saved as pickle) that should be loaded
        If None, a new data_storage is created
    save_data_storage_path: str, default: None
        Path to store created (or loaded) DataStorage instance (will be saved as pickle)
        If None, a DataStorage instance will not be saved
    total_look_back: int
        number of periods that should be included in historical lookback
    time_intervals: list(pd.Interval)
        list of time intervals that are included
    time_intervals_medic: list(pd.Interval)
        list of time intervals that can be used for medic method
    latitude_intervals: pd.interval_range
        latitude values for vertical area divisions
    longitude_intervals: pd.interval_range
        longitude values for horizontal area divisions
    nrRows: int
        number of horizontal area divisions (=len(longitude_intervals))
    nrCols: int
        number of vertical area divisions (=len(latitude_intervals))
    calldata: str
        name of dataset, supported: 'Seattle'
    filter_dict: dict(key: column name in call data, values: column values that should be filtered, e.g. {'Type': ['Medic Response']})
        dictionary including column names and values (in calldata) to filter (e.g. types of calls, call priority, etc.)
    features_csv: str
        name of csv file in which feature information is saved (is later read into features dataframe)
        Must include columns: Feature, Time, Input_Shape, One_Hot_Encoding, Concatenation, Transpose, Include, Category
    shorten_one_hot: bool, optional, default: True
        If false, vector of length 24/7/12 is returned for hour/day/month, correspondingly
        If true, vector length is adapted to the number of distinct time values
    data_types: List[types.DataType], default: None
        data type for which data should be created (e.g. data structure for CNN differs from structure required by MLP)

    Returns:
    data_storage: DataStorage
        DataStorage instance with saved data for provided models
    """
    if load_data_storage_path is not None and os.path.exists(load_data_storage_path):
        data_storage = __load_data_storage(load_data_storage_path)

    else:
        data_storage = __init_data_storage(
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
            shorten_one_hot,
            data_types,
        )

    if save_data_storage_path:
        __save_data_storage(data_storage, save_data_storage_path)

    return data_storage


def __save_data_storage(data_storage: DataStorage, path: str) -> None:
    """Saves DataStorage instance

    Parameters
    data_storage: DataStorage
        DataStorage to be saved
    save_data_storage_path: str
        Path to store provided DataStorage instance (will be saved as pickle)
    """
    with open(path, "wb") as file:
        pickle.dump(data_storage, file)


def __load_data_storage(path: str) -> DataStorage:
    """Loads DataStorage instance

    Parameters
    path: str
        Path to DataStorage instance to be loaded

    Returns:
    data_storage: DataStorage
        DataStorage instance stored at path
    """
    try:
        with open(path, "rb") as file:
            ds = pickle.load(file)
    except:
        ds = pd.read_pickle(path)
    return ds
