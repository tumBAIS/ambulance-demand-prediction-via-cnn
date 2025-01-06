from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from Data_Handler.data_storage import DataStorage

FIXED_SETTINGS_FILE = "./Settings/MEDIC/settings_medic_fixed.yaml"
RESULTS_FILE_NAME = "medic_results.yaml"

# pylint: disable=no-member, too-many-instance-attributes
class MedicMethod:
    def __init__(self):

        # Set statically defined hyperparameters via yaml file
        # replace such that files can be named by id
        self.id = str(datetime.today()).replace(":", "-")
        self.id = self.id.replace(" ", "-")
        with open(FIXED_SETTINGS_FILE, "r") as stream:
            try:
                fixed_settings = yaml.safe_load(stream)
                for key, value in fixed_settings.items():
                    setattr(self, key, value)
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def get_lags(
        time_int_per_day: int = 1, nr_weeks: int = 4, nr_years: int = 5
    ) -> List[int]:
        """Generates the indices of preceding periods to be considered in calculations

        Parameters:
        time_int_per_day: int
            number of periods that fit in one day
        nr_weeks: int
            number of preceding weeks we consider for calculations
        nr_years: int
            number of preceding years we consider for calculations
        """

        lags_past_weeks = [t for t in range(7, 7 * nr_weeks + 1, 7)]
        lags_past_years = [
            t + (52 * 7) * year for t in lags_past_weeks for year in range(nr_years)
        ]
        lags_in_time_intervals = [t * time_int_per_day for t in lags_past_years]
        lags_in_time_intervals.sort()
        return lags_in_time_intervals

    @staticmethod
    def get_time_int_per_day(time_interval: str) -> int:
        """Calculates the number of periods that fit in one day

        Parameters:
        time_interval: str
            time interval we consider, e.g., "8H" or "1D"

        """
        if time_interval.endswith("H"):
            nr = 24 / int(time_interval.removesuffix("H"))
        elif time_interval.endswith("D"):
            nr = 1 / int(time_interval.removesuffix("D"))
        else:
            raise NotImplementedError()
        if nr.is_integer():
            return int(nr)
        else:
            raise Exception("Number of intervals per day must be integer")


def run_medic_method(
    data_storage: DataStorage,
    time_intervals: List[pd.Interval],
    time_intervals_medic: List[pd.Interval],
    nr_weeks: int,
    nr_years: int,
    nrRows: int,
    nrCols: int,
    time_interval: str,
    output_path: str,
) -> None:
    """Runs medic method and dumps results

    Parameters:
    data_storage: DataStorage
        Data storage instance containing required data
    time_intervals: List[pd.Interval]
        list of time intervals that are included
    time_intervals_medic: List[pd.Interval]
        list of time intervals that are included in medic method
    nr_weeks: int
        number of preceding weeks we consider for calculations
    nr_years: int
        number of preceding years we consider for calculations
    nrRows: int
        number of horizontal area divisions (=len(longitude_intervals))
    nrCols: int
        number of vertical area divisions (=len(latitude_intervals))
    time_interval: str
        time interval we consider, e.g., "8H" or "1D"
    output_path: str
        Path where to store results
    """

    # generate MLP
    medic = MedicMethod()

    time_series_per_subregion = data_storage.get_time_series_medic()

    if time_series_per_subregion is None:
        raise ValueError("Please provide X or y or store data in data_storage.")

    nr_time_intervals_considered = len(time_intervals)
    nr_time_intervals_total = len(time_intervals_medic)
    nr_test_instances = int(nr_time_intervals_considered * medic.test_size)

    series_test_indices = range(
        nr_time_intervals_total - nr_test_instances, nr_time_intervals_total
    )

    predictions = np.zeros((nrRows, nrCols, nr_test_instances))
    y = np.zeros((nrRows, nrCols, nr_test_instances))

    for r in range(nrRows):
        for c in range(nrCols):
            series = time_series_per_subregion[r][c]
            for idx, t in enumerate(series_test_indices):
                historic_demand_r_c = np.array([])
                nr_time_intervals_per_day = medic.get_time_int_per_day(time_interval)
                lags = medic.get_lags(
                    time_int_per_day=nr_time_intervals_per_day,
                    nr_weeks=nr_weeks,
                    nr_years=nr_years,
                )
                for _, l in enumerate(lags):
                    if (t - l) > 0:  # only go back as far as we have data
                        historic_demand_r_c = np.append(
                            historic_demand_r_c, series[t - l]
                        )

                predictions[r][c][idx] = np.mean(historic_demand_r_c)
                y[r][c][idx] = series[t]

    MSE = (1 / y.size) * np.sum(np.square(y - predictions), axis=None)

    non_zero_indices = y != 0
    MSE_non_zero_demands = (1 / y[non_zero_indices].size) * np.sum(
        np.square(y[non_zero_indices] - predictions[non_zero_indices]), axis=None
    )

    zero_indices = y == 0
    MSE_zero_demands = (1 / y[zero_indices].size) * np.sum(
        np.square(y[zero_indices] - predictions[zero_indices]), axis=None
    )

    results = {
        "MSE": MSE.item(),
        "MSE_non_zero_demands": MSE_non_zero_demands.item(),
        "MSE_zero_demands": MSE_zero_demands.item(),
    }

    results_file_name = f"{medic.id}_{RESULTS_FILE_NAME}"
    with open(output_path + results_file_name, "w") as file:
        yaml.dump(results, file)
