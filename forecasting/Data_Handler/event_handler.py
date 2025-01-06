from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pygeoj
from shapely.geometry import Polygon

DATA_FILE_PATH_EVENTS = "./Data/Special_Events_Permits_Seattle.csv"
DATA_FILE_PATH_NEIGHBORHOODS = "./Data/Seattle_Neighborhoods.geojson"

COLUMN_LABEL_STATUS = "Status"
COLUMN_LABEL_START_DATE = "Event_Start_Date"
COLUMN_LABEL_END_DATE = "Event_End_Date"
COLUMN_LABEL_NEIGHBORHOOD = "Neighborhood"
COLUMN_LABEL_ATTENDANCE = "Attendance"

COLUMNS_RENAME_DICT = {
    "Permit Status": COLUMN_LABEL_STATUS,
    "Event Start Date": COLUMN_LABEL_START_DATE,
    "Event End Date": COLUMN_LABEL_END_DATE,
    "Event Location - Neighborhood": COLUMN_LABEL_NEIGHBORHOOD,
    "Attendance": COLUMN_LABEL_ATTENDANCE,
}


class EventCube:
    def __init__(
        self,
        time_intervals: List[pd.Interval],
        latitude_intervals: pd.interval_range,
        longitude_intervals: pd.interval_range,
        nrRows: int,
        nrCols: int,
    ) -> None:
        """Generates event cube

        Parameters:
        time_intervals: List[pd.Interval]
            list of time periods that are examined
        latitude_intervals: pd.interval_range
            latitude values for vertical area divisions
        longitude_intervals: pd.interval_range
            longitude values for horizontal area divisions
        nrRows: int
            number of horizontal area divisions
        nrCols: int
            number of vertical area divisions
        """
        # Generate event data set based on temporal and spatial intervals
        # Done here, such that it must be calculated only once
        jsonfile = self.__read_neighborhoods()
        neighborhood_polygons = self.__generate_neighborhood_polygons(jsonfile)
        area_polygons = self.__generate_area_polygons(
            latitude_intervals, longitude_intervals
        )
        neighborhood_area_intersects = self.__generate_neighborhood_area_intersec(
            nrRows, nrCols, area_polygons, neighborhood_polygons
        )
        events_df = self.__get_event_data_seattle(neighborhood_polygons)
        event_cube = self.__generate_event_cube(
            events_df, time_intervals, neighborhood_area_intersects, nrRows, nrCols
        )
        event_cube = event_cube / np.max(event_cube)  # normalize
        self.__event_cube = event_cube

    def get_event_cube(self) -> np.ndarray:
        """Returns cube with events (nr. of attendees per area), assumption: if event takes
        place in multiple neighborhoods, we assume that in each neighborhood there are an equal number of people

        Returns:
        event_cube: np.ndarray
            3D array with events (nr. of attendees per area)
        """
        return self.__event_cube

    def __get_event_data_seattle(
        self, neighborhood_polygons: Dict[str, Polygon]
    ) -> pd.DataFrame:
        """Generates dataframe with event data

        Parameters:
        neighborhood_polygons: Dict[str, Polygon]
            key: neighborhood name (s_hood), value: polygon of neighborhood

        Returns:
        events_df: pd.DataFrame
            Dataframe with events data
        """

        events_df = pd.read_csv(DATA_FILE_PATH_EVENTS, sep=",")

        # Rename columns to avoid spaces
        events_df.rename(
            columns=COLUMNS_RENAME_DICT,
            inplace=True,
        )

        # Remove all cancelled events and events for which no location is given
        events_df = events_df.drop(events_df[events_df.Status == "Cancelled"].index)
        events_df = events_df[events_df[COLUMN_LABEL_NEIGHBORHOOD].notna()]

        # Downtown corresponds to several neighborhood, Capitol Hill corresponds to Broadway
        events_df[COLUMN_LABEL_NEIGHBORHOOD].replace(
            {
                "Downtown": "Pioneer Square, International District, Yesler Terrace, Central Business District, Pike-Market, Belltown",
                "Capitol Hill": "Broadway",
            },
            inplace=True,
        )

        # Correct column types (dates)
        events_df[COLUMN_LABEL_ATTENDANCE] = events_df[COLUMN_LABEL_ATTENDANCE].fillna(
            0
        )
        events_df[COLUMN_LABEL_ATTENDANCE] = events_df[COLUMN_LABEL_ATTENDANCE].astype(
            int
        )

        events_df[COLUMN_LABEL_START_DATE] = events_df.Event_Start_Date.apply(
            lambda x: pd.to_datetime(x).strftime("%m/%d/%Y")[:]
        )
        events_df[COLUMN_LABEL_END_DATE] = events_df.Event_End_Date.apply(
            lambda x: pd.to_datetime(x).strftime("%m/%d/%Y")[:]
        )

        events_df[COLUMN_LABEL_START_DATE] = pd.to_datetime(
            events_df[COLUMN_LABEL_START_DATE], format="%m/%d/%Y"
        )
        events_df[COLUMN_LABEL_END_DATE] = pd.to_datetime(
            events_df[COLUMN_LABEL_END_DATE], format="%m/%d/%Y"
        )
        events_df[COLUMN_LABEL_START_DATE] = events_df.apply(
            lambda row: row.Event_Start_Date.date(), axis=1
        )
        events_df[COLUMN_LABEL_END_DATE] = events_df.apply(
            lambda row: row.Event_End_Date.date(), axis=1
        )

        # Add column for each neighborhood
        nr_neighborhoods = len(list(neighborhood_polygons.keys()))
        for s_hood in neighborhood_polygons.keys():
            events_df[s_hood] = events_df[COLUMN_LABEL_NEIGHBORHOOD].str.contains(
                s_hood
            )
            events_df[s_hood] = events_df[s_hood].astype(int)

        # Check in how many neighborhoods events take place, remove events for which no neighborhood was found
        nr_neighborhoods_col = "NrNeighborhoods"
        events_df[nr_neighborhoods_col] = events_df.iloc[:, -nr_neighborhoods:].sum(
            axis=1
        )
        events_df = events_df.drop(
            events_df[events_df[nr_neighborhoods_col] == 0].index
        )

        # Get percentage share that is in each neighborhood (often, event is taking place in mutiple neighborhoods)
        # Assumption: in each neighborhood, similar share of attendance

        # new pandas version
        events_df[events_df.columns[-(nr_neighborhoods - 1) : -1]] = events_df[
            events_df.columns[-(nr_neighborhoods - 1) : -1]
        ].div(events_df[nr_neighborhoods_col], axis=0)
        events_df.pop(nr_neighborhoods_col)

        # Multiply attendance with percentage share in each neighborhood
        events_df.iloc[:, -nr_neighborhoods:-1] = events_df.iloc[
            :, -nr_neighborhoods:-1
        ].multiply(events_df[COLUMN_LABEL_ATTENDANCE], axis=0)

        # Convert columns into rows such that each neighborhood and event has its own row
        id_vars = list(events_df.columns)[:-nr_neighborhoods]
        events_df = events_df.melt(
            id_vars=id_vars, var_name="s_hood", value_name="Attendance_Share"
        )

        # Delete all rows in which a neighborhood has no event
        events_df = events_df.drop(events_df[events_df["Attendance_Share"] == 0].index)

        return events_df

    def __read_neighborhoods(self) -> Any:
        """Loads geojson file

        Returns:
        GeojsonFile
            A GeojsonFile instance
        """
        return pygeoj.load(filepath=DATA_FILE_PATH_NEIGHBORHOODS)

    def __generate_area_polygons(
        self,
        latitude_intervals: pd.interval_range,
        longitude_intervals: pd.interval_range,
    ) -> Dict[int, Polygon]:
        """Generates dictionary with area nr. (key) and the corresponding polygon (value)

        Parameters:
        latitude_intervals: pd.interval_range
            latitude values for vertical area divisions
        longitude_intervals: pd.interval_range
            longitude values for horizontal area divisions

        Returns:
        area_polygons: Dict[int, Polygon]
            dictionary including area nr. (key) and corresponding polygon (value)
        """

        area_polygons = {}
        i = 0
        for lat in latitude_intervals:
            for lon in longitude_intervals:
                area_polygons[i] = Polygon(
                    [
                        (lon.left, lat.left),
                        (lon.left, lat.right),
                        (lon.right, lat.right),
                        (lon.right, lat.left),
                    ]
                )
                i += 1
        return area_polygons

    def __generate_neighborhood_polygons(self, jsonfile: Any) -> Dict[str, Polygon]:
        """Generates dictionary with neighborhood name (key) and the corresponding polygon (value)

        Parameters:
        jsonfile:
            GeojsonFile instance including neighbourhood information

        Returns:
        neighborhood_polygons: Dict[str, Polygon]
            dictionary including neighbourhood name (key) and corresponding polygon (value)
        """
        neighborhood_polygons = {}
        for f in range(len(jsonfile)):
            s_hood = jsonfile.get_feature(f).properties["S_HOOD"]
            coordinates_list = jsonfile.get_feature(f).geometry.coordinates[0]
            coordinates_tuples = [tuple(l) for l in coordinates_list]
            if s_hood not in ("OOO", " "):
                neighborhood_polygons[s_hood] = Polygon(coordinates_tuples)
        return neighborhood_polygons

    def __generate_neighborhood_area_intersec(
        self,
        nrRows: int,
        nrCols: int,
        area_polygons: Dict[int, Polygon],
        neighborhood_polygons: Dict[str, Polygon],
    ) -> Dict[str, np.ndarray]:
        """Generates dictionary with neighborhood name (key) and a matrix including the percentage
        neighborhood share which is in each area (square) (value)

        Parameters:
        nrRows: int
            nrRows whole area is divided to
        nrCols: int
            nrCols whole area is divided to
        area polygons: Dict[int, Polygon]
            dict with area id (int) as key, and its polygon as values
        neighborhood polygon: Dict[str, Polygon]
            dict with neighborhood name (string) as key, and its polygon as value

        Returns:
        neighborhood_area_intersec: Dict[str, np.ndarray]
            dictionary including neighbourhood name (key) and a matrix including the percentage
            neighborhood share which is in each area (square) (value)
        """
        neighborhood_area_intersec = {}
        area_ids = list(area_polygons.keys())
        for s_hood in neighborhood_polygons.keys():
            # Matrix to store percentage how "much" neighbourhood is in each area
            neighborhood_area_intersec[s_hood] = np.zeros((nrRows, nrCols))
            # Polygon of neighborhood
            neighborhood_polygon = neighborhood_polygons[s_hood]
            a = 0
            for r in range(nrRows):
                for c in range(nrCols):
                    # Calculate 1) intersection between neighborhood & area a 2) percentage of neighborhood in area a
                    intersect = neighborhood_polygon.intersection(
                        area_polygons[area_ids[a]]
                    ).area
                    neighborhood_area_intersec[s_hood][r, c] = (
                        intersect / neighborhood_polygon.area
                    )
                    a += 1
        return neighborhood_area_intersec

    def __generate_event_cube(
        self,
        events_df: pd.DataFrame,
        time_intervals: List[pd.Interval],
        neighborhood_area_intersects: Dict[str, np.ndarray],
        nrRows: int,
        nrCols: int,
    ) -> np.ndarray:
        """Generates cube with events (nr. of attendees per area), assumption: if event takes
        place in multiple neighborhoods, we assume that in each neighborhood there is an equal number of people

        Parameters:
        events_df: pd.DataFrame
            Dataframe with events data
        time_intervals: List[pd.Interval]
            list of time periods that are examined
        neighborhood_area_intersects: Dict[str, np.ndarray]
            dictionary with neighborhood name (key) and a matrix including the percentage
            neighborhood share which is in each area (square) (value)
        nrRows: int
            nrRows whole area is divided to
        nrCols: int
            nrCols whole area is divided to

        Returns:
        event_cube: np.ndarray
            3D-array with event data including the number of participants per square
        """

        events_df = events_df[
            events_df[COLUMN_LABEL_END_DATE]
            > pd.Timestamp(time_intervals[0].left).date()
        ]
        events_df = events_df[
            events_df[COLUMN_LABEL_START_DATE]
            < pd.Timestamp(time_intervals[len(time_intervals) - 1].right).date()
        ]

        event_cube = np.zeros((nrRows, nrCols, len(time_intervals)))
        for i, t in enumerate(time_intervals):
            x = events_df[
                (events_df[COLUMN_LABEL_START_DATE] <= pd.Timestamp(t.left).date())
                & (events_df[COLUMN_LABEL_END_DATE] >= pd.Timestamp(t.left).date())
            ]
            for _, row in x.iterrows():
                event_cube[:, :, i] += (
                    neighborhood_area_intersects[row["s_hood"]]
                    * row["Attendance_Share"]
                )

        return event_cube
