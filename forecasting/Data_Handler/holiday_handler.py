from datetime import timedelta
from typing import List

import pandas as pd
from Data_Handler.feature_names import Feature_wo_Time

DATA_FILE_PATH_PUBLIC_HOLIDAYS = "./Data/Public_Holidays_Seattle.csv"
DATA_FILE_PATH_SCHOOL_HOLIDAYS = "./Data/School_Holidays_Seattle.csv"

COLUMN_LABELS_MAP = {
    "Public_Holidays": Feature_wo_Time.PUBLIC_HOLIDAYS.name,
    "School_Holidays": Feature_wo_Time.SCHOOL_HOLIDAYS.name,
}
PUBLIC_HOLIDAY_LABEL_DATE = "Date"
SCHOOL_HOLIDAY_LABEL_FROM_DATE = "From_Date"
SCHOOL_HOLIDAY_LABEL_TO_DATE = "To_Date"


class HolidayHandler:
    def __init__(self, time_intervals: List[pd.Interval]):
        """
        Parameters:
            time_intervals: List[pd.Interval]
                List of time intervals considered
        """

        self.__holiday_data = self.__generate_holiday_data(time_intervals)

    def get_holiday_data(self) -> pd.DataFrame:
        return self.__holiday_data

    # Public Holidays Data Seattle
    def __generate_holiday_data(
        self, time_intervals: List[pd.Interval]
    ) -> pd.DataFrame:
        """Reads Seattle's Holiday Data from csv-sheets.

        Parameters:
            time_intervals: List[pd.Interval]
                List of time intervals considered

        Returns:
            holiday_data: pd.DataFrame
                Dataframe with the daily holiday data at each TimeInterval Start
                (boolean: True: holiday/ False: no holiday)

        """
        # Transform time intervals into a data frame and extract only starting date of each period
        time_intervals_df = pd.DataFrame(
            list(time_intervals.left), columns=["TimeIntervalStart"]
        )
        time_intervals_df["Date"] = time_intervals_df.apply(
            lambda row: row["TimeIntervalStart"].date(), axis=1
        )

        # Read public holiday data and correct column types
        public_holidays_seattle_df = self.__generate_public_holiday_df()

        # Read school holiday data and correct column types
        school_holidays_seattle_df = self.__generate_school_holiday_df()

        school_hol_list = []
        for d in time_intervals_df["Date"]:
            count = school_holidays_seattle_df[
                (
                    school_holidays_seattle_df[SCHOOL_HOLIDAY_LABEL_FROM_DATE]
                    - timedelta(days=1)
                    < d
                )
                & (
                    school_holidays_seattle_df[SCHOOL_HOLIDAY_LABEL_TO_DATE]
                    + timedelta(days=1)
                    > d
                )
            ].count()
            if count[0] > 0:
                school_hol_list.append(1)
            else:
                school_hol_list.append(0)

        time_intervals_df["School_Holidays"] = school_hol_list

        # Merge holiday data with time intervals
        holiday_data = pd.merge(
            time_intervals_df, public_holidays_seattle_df, how="left", on="Date"
        )

        # Replace public holidays with 1, no holidays with 0
        holiday_data["Public_Holidays"] = (
            holiday_data["Public_Holidays"].notnull().astype("int")
        )

        holiday_data.rename(
            columns=COLUMN_LABELS_MAP,
            inplace=True,
        )

        return holiday_data

    def __generate_school_holiday_df(self) -> pd.DataFrame:
        """Reads school holidays from csv and transforms data types

        Returns:
            school_holidays_seattle_df: pd.DataFrame
                Dataframe including school holidays
        """
        school_holidays_seattle_df = pd.read_csv(
            DATA_FILE_PATH_SCHOOL_HOLIDAYS, sep=";"
        )
        school_holidays_seattle_df[SCHOOL_HOLIDAY_LABEL_FROM_DATE] = pd.to_datetime(
            school_holidays_seattle_df[SCHOOL_HOLIDAY_LABEL_FROM_DATE],
            format="%d.%m.%Y",
        )
        school_holidays_seattle_df[SCHOOL_HOLIDAY_LABEL_TO_DATE] = pd.to_datetime(
            school_holidays_seattle_df[SCHOOL_HOLIDAY_LABEL_TO_DATE], format="%d.%m.%Y"
        )
        school_holidays_seattle_df[
            SCHOOL_HOLIDAY_LABEL_FROM_DATE
        ] = school_holidays_seattle_df.apply(
            lambda row: row[SCHOOL_HOLIDAY_LABEL_FROM_DATE].date(), axis=1
        )
        school_holidays_seattle_df[
            SCHOOL_HOLIDAY_LABEL_TO_DATE
        ] = school_holidays_seattle_df.apply(
            lambda row: row[SCHOOL_HOLIDAY_LABEL_TO_DATE].date(), axis=1
        )

        return school_holidays_seattle_df

    def __generate_public_holiday_df(self) -> pd.DataFrame:
        """Reads public holidays from csv and transforms data types

        Returns:
            public_holidays_seattle_df: pd.DataFrame
                Dataframe including public holidays
        """
        public_holidays_seattle_df = pd.read_csv(
            DATA_FILE_PATH_PUBLIC_HOLIDAYS, sep=";"
        )
        public_holidays_seattle_df[PUBLIC_HOLIDAY_LABEL_DATE] = pd.to_datetime(
            public_holidays_seattle_df[PUBLIC_HOLIDAY_LABEL_DATE], format="%d.%m.%Y"
        )
        public_holidays_seattle_df[
            PUBLIC_HOLIDAY_LABEL_DATE
        ] = public_holidays_seattle_df.apply(
            lambda row: row[PUBLIC_HOLIDAY_LABEL_DATE].date(), axis=1
        )

        return public_holidays_seattle_df
