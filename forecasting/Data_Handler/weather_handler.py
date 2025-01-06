from datetime import date
from typing import List

import pandas as pd
from Data_Handler.feature_names import Feature_wo_Time

DATA_FILE_PATH_WEATHER = "./Data/Weather_Seattle.csv"

COLUMN_LABELS_MAP = {
    "Min_Temp_F": Feature_wo_Time.MIN_TEMP_F.name,
    "Avg_Temp_F": Feature_wo_Time.AVG_TEMP_F.name,
    "Max_Temp_F": Feature_wo_Time.MAX_TEMP_F.name,
    "Min_Wind_Speed_mph": Feature_wo_Time.MIN_WIND_SPEED_MPH.name,
    "Avg_Wind_Speed_mph": Feature_wo_Time.AVG_WIND_SPEED_MPH.name,
    "Max_Wind_Speed_mph": Feature_wo_Time.MAX_WIND_SPEED_MPH.name,
    "Min_Humidity_Percent": Feature_wo_Time.MIN_HUMIDITY_PERCENT.name,
    "Avg_Humidity_Percent": Feature_wo_Time.AVG_HUMIDITY_PERCENT.name,
    "Max_Humidity_Percent": Feature_wo_Time.MAX_HUMIDITY_PERCENT.name,
    "Min_Dew_Point_F": Feature_wo_Time.MIN_DEW_POINT_F.name,
    "Avg_Dew_Point_F": Feature_wo_Time.AVG_DEW_POINT_F.name,
    "Max_Dew_Point_F": Feature_wo_Time.MAX_DEW_POINT_F.name,
    "Min_Pressure_Hg": Feature_wo_Time.MIN_PRESSURE_HG.name,
    "Avg_Pressure_Hg": Feature_wo_Time.AVG_PRESSURE_HG.name,
    "Max_Pressure_Hg": Feature_wo_Time.MAX_PRESSURE_HG.name,
    "Total_Precipitation_intensity": Feature_wo_Time.TOTAL_PRECIPITATION_INTENSITY.name,
}


class WeatherHandler:
    def __init__(self, time_intervals: List[pd.Interval]):

        self.__weather_data = self.__generate_weather_data(time_intervals)

    def get_weather_data(self) -> pd.DataFrame:
        return self.__weather_data

    def __generate_weather_data(
        self, time_intervals: List[pd.Interval]
    ) -> pd.DataFrame:
        """Reads Seattle Weather Data from csv-sheet.
        The daily data is taken from the date at which the time interval has started.

        Parameters:
        time_intervals: List[pd.Interval]
            List of time intervals considered

        Returns:
        weather_data: pd.DataFrame
            Dataframe with the daily weather data at each TimeInterval Start
            (Temperature, dew Point, Humidity, Wind Speed, Pressure, Precipitation intensity)

        """
        # Transform time intervals into a data frame and extract only starting date of each period
        time_intervals_df = pd.DataFrame(
            list(time_intervals.left), columns=["TimeIntervalStart"]
        )
        time_intervals_df["Date"] = time_intervals_df.apply(
            lambda row: row.TimeIntervalStart.date(), axis=1
        )

        # Read weather data and correct column types
        weather_seattle_df = pd.read_csv(DATA_FILE_PATH_WEATHER, sep=";")
        weather_seattle_df["Date"] = weather_seattle_df.apply(
            lambda row: date(int(row.Year), int(row.Month), int(row.Day)), axis=1
        )

        weather_seattle_df.rename(columns=COLUMN_LABELS_MAP, inplace=True)

        # Merge weather data with time intervals
        weather_data = pd.merge(
            time_intervals_df, weather_seattle_df, how="left", on="Date"
        )

        weather_data = self.__preprocess_weather_data_seattle(weather_data)

        return weather_data

    def __preprocess_weather_data_seattle(
        self, weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms weather data such that weather data is in range [0,1]

        Returns:
        weather_data: pd.DataFrame
            Dataframe with the daily weather data at each TimeInterval Start
            (Temperature, Dew Point, Humidity, Wind Speed, Pressure, Precipitation intensity)

        Returns:
        weather_data: pd.DataFrame
            Scaled Dataframe with the daily weather data at each TimeInterval Start
            (Temperature, Dew Point, Humidity, Wind Speed, Pressure, Precipitation intensity)
        """

        weather_features = [
            "MIN_TEMP_F",
            "AVG_TEMP_F",
            "MAX_TEMP_F",
            "MIN_WIND_SPEED_MPH",
            "AVG_WIND_SPEED_MPH",
            "MAX_WIND_SPEED_MPH",
            "MIN_HUMIDITY_PERCENT",
            "AVG_HUMIDITY_PERCENT",
            "MAX_HUMIDITY_PERCENT",
            "MIN_DEW_POINT_F",
            "AVG_DEW_POINT_F",
            "MAX_DEW_POINT_F",
            "MIN_PRESSURE_HG",
            "AVG_PRESSURE_HG",
            "MAX_PRESSURE_HG",
            "TOTAL_PRECIPITATION_INTENSITY",
        ]

        for f in weather_features:

            min_f = min(weather_data[f])
            range_f = max(weather_data[f]) - min_f
            weather_data[f] = (weather_data[f] - min_f) / range_f

        return weather_data
