from typing import Dict, List

import numpy as np
import pandas as pd
from sodapy import Socrata

DATA_FILE_PATH_AMBULANCE_DEMAND = "./Data/Seattle_Real_Time_Fire_911_Calls.csv"
DATASET_NAME_SEATTLE = "Seattle"
DF_LABEL_DATETIME = "Datetime"
DF_LABEL_LONGITUDE = "Longitude"
DF_LABEL_LATITUDE = "Latitude"
DF_LABEL_INCIDENT_NR = "Incident Number"
DF_LABEL_CALL_COUNTS = "count"


class HeatmapCube:
    def __init__(self, dataset_name: str):
        """
        Parameters:
            dataset: str
                Name of dataset (supported: "Seattle")
        """
        self.__dataset_name = dataset_name

    def load_data(self) -> pd.DataFrame:
        """Loads data from csv

        Returns:
            data: pd.DataFrame
                DataFrame with loaded data
        """
        if self.__dataset_name == DATASET_NAME_SEATTLE:
            data = self.__load_data_seattle()
        else:
            ValueError(
                "Dataset ",
                self.__dataset_name,
                " is unknown. Please implement function to load.",
            )
        return data

    def get_heatmap_cube(
        self,
        filter_dict: Dict[str, List[str]],
        time_intervals: List[pd.Interval],
        latitude_intervals: pd.interval_range,
        longitude_intervals: pd.interval_range,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Generates callcube, normalizes data if <normalize> = True and updates <self.__heatmap_cube>

        Parameters:
            filter_dict: Dict[str, List[str]]
                Dictionary: key: column name to be filtered,
                value: column values that should be filtered by (i.e. included)
            time_intervals: List[pd.Interval]
                List with time intervals by which data should be grouped
            latitude_intervals: pd.interval_range
                List of latitudes by which data should be grouped
            longitude_intervals: pd.interval_range
                List of longitudes by which data should be grouped
            normalize: bool, default: False
                True, if call data should be normalized (call data = call data / max(call data))

        Returns:
            self.__heatmap_cube: np.ndarray
                Callcube, normalized if <normalize> = True
        """

        data = self.load_data()
        data = self.__filter_data(data, filter_dict)
        grouped_df = self.__get_counts(
            data, time_intervals, latitude_intervals, longitude_intervals
        )
        call_cube = self.__generate_callcube(
            grouped_df, latitude_intervals, longitude_intervals
        )
        if normalize:
            self.__heatmap_cube = call_cube / (np.max(call_cube))
        else:
            self.__heatmap_cube = call_cube

        return self.__heatmap_cube

    def __load_data_seattle(self) -> pd.DataFrame:
        """Loads seattle dataset via csv

        Returns:
            einsatz_data: pd.DataFrame
                DataFrame with incident data
        """

        einsatz_data = pd.read_csv(DATA_FILE_PATH_AMBULANCE_DEMAND, sep=",")

        einsatz_data[DF_LABEL_DATETIME] = pd.to_datetime(
            einsatz_data[DF_LABEL_DATETIME], format="%m/%d/%Y %I:%M:%S %p"
        )

        einsatz_data[DF_LABEL_LONGITUDE] = einsatz_data[DF_LABEL_LONGITUDE].astype(
            float
        )
        einsatz_data[DF_LABEL_LATITUDE] = einsatz_data[DF_LABEL_LATITUDE].astype(float)

        einsatz_data.dropna(inplace=True)

        return einsatz_data

    def __get_counts(
        self,
        data: pd.DataFrame,
        time_intervals: List[pd.Interval],
        int_lat: pd.interval_range,
        int_lon: pd.interval_range,
    ):
        """
        Returns call data counts grouped by time intervals, longitude and latitude intervals

        Returns a dataframe grouped by time intervals, longitude and latitude intervals and
        counts the number of unique entries in <DF_LABEL_INCIDENT_NR> or,
        if <DF_LABEL_INCIDENT_NR> is None, the number of rows

        Parameters:
            data: pd.DataFrame
                Dataframe with data to be grouped, must contain the following columns:
                - Datetime: Including time by which data should be grouped (e.g. date and time of incident)
                - Latitude: Latitude by which data should be grouped (e.g. latitude of incident)
                - Longitude: Latitude by which data should be grouped (e.g. latitude of incident)

            time_intervals: List[pd.Interval]
                List with time intervals by which data should be grouped

            int_lat: pd.interval_range
                List of latitudes by which data should be grouped

            int_lon: pd.interval_range
                List of longitudes by which data should be grouped

        Returns:
            grouped_df: pd.Dataframe
                Dataframe containing counted values in column <DF_LABEL_CALL_COUNTS>
                grouped by spatial area (longitudes and latitudes) and time intervals

        """
        # Reverse sorting (descending latitude), latitude (rows) must be sorted
        # descending such that ([0,:]) has a higher latitude than([5,:])
        int_lat = int_lat[::-1]
        grouped_df = (
            data.groupby(
                [
                    pd.cut(data.Datetime, time_intervals),
                    pd.cut(data.Latitude, int_lat),
                    pd.cut(data.Longitude, int_lon),
                ]
            )[DF_LABEL_INCIDENT_NR]
            .count()
            .reset_index(name=DF_LABEL_CALL_COUNTS)
        )
        return grouped_df

    def __generate_callcube(
        self,
        grouped_df: pd.DataFrame,
        int_lat: List[pd.Interval],
        int_lon: List[pd.Interval],
    ) -> np.ndarray:
        """Gets counts in dataframe column <LABEL_CALL_COUNTS> and transforms counts into np.ndarray.
        Prerequisite: dataframe must be grouped according to time, longitudes and latitude intervals

        Parameters:
        grouped_df: pd.DataFrame
            Grouped dataframe. Must include column <LABEL_CALL_COUNTS> including call counts per time interval and square
        int_lat: List[pd.Interval]
            List with latitude intervals
        int_lon: List[pd.Interval]
            List with longitude intervals

        Returns:
        call_cube: np.ndarray
            3D array with call counts
        """
        counts = grouped_df[DF_LABEL_CALL_COUNTS].to_numpy()
        call_cube = counts.reshape(-1, len(int_lat), len(int_lon)).transpose(1, 2, 0)
        return call_cube

    def __filter_data(
        self, data: pd.DataFrame, filter_dict: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Filters data according to filter_dict

        Parameters:
        data: pd.Dataframe
            Dataframe to be filtered
        filter_dict: dict
            Dictionary including filter information:
            keys: column names to be filtered, values: column values to be filtered (i.e. included)

        Returns
        data: pd.Dataframe
            Filtered dataframe
        """
        if isinstance(filter_dict, dict):
            for col, values in filter_dict.items():
                data = data[data[col].isin(values)]
            return data
        raise ValueError(
            "filter_dict must be of type ´dict´ but ist of type ", type(filter_dict)
        )
