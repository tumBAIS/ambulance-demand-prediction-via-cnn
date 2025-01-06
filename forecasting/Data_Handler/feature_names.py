from enum import Enum

FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME = "Feature"
FEATURE_SETTINGS_LABEL_TIME = "Time"
FEATURE_SETTINGS_LABEL_INPUT_SHAPE = "Input_Shape"
FEATURE_SETTINGS_LABEL_ONE_HOT_ENCODING = "One_Hot_Encoding"
FEATURE_SETTINGS_LABEL_CONCATENATION = "Concatenation"
FEATURE_SETTINGS_LABEL_TRANSPOSE = "Transpose"
FEATURE_SETTINGS_LABEL_INCLUSION = "Include"
FEATURE_SETTINGS_LABEL_CATEGORY = "Category"
FEATURE_SETTINGS_LABEL_FEATURE_W_TIME = "Feature_w_Time"
FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME = "Feature_Name"


class Feature_w_Time(Enum):
    AMBULANCE_DEMAND_HISTORY = "Historic Ambulance Demand (periods [T-L, t-1])"
    HOURS_PREDICTION = "Hour (period t)"
    WEEKDAYS_PREDICTION = "Weekday (period t)"
    MONTHS_PREDICTION = "Month (period t)"
    SCHOOL_HOLIDAYS_PREDICTION = "School Holidays (period t)"
    PUBLIC_HOLIDAYS_PREDICTION = "Public Holidays (period t)"
    EVENTS_PREDICTION = "Events (period t)"
    MAX_TEMP_F_PREDICTION = "Max. Temperature [F] (period t)"
    AVG_TEMP_F_PREDICTION = "Avg. Temperature [F] (period t)"
    MIN_TEMP_F_PREDICTION = "Min. Temperature [F] (period t)"
    MAX_WIND_SPEED_MPH_PREDICTION = "Max. Wind Speed [mph] (period t)"
    AVG_WIND_SPEED_MPH_PREDICTION = "Avg. Wind Speed [mph] (period t)"
    MIN_WIND_SPEED_MPH_PREDICTION = "Min. Wind Speed [mph] (period t)"
    MAX_HUMIDITY_PERCENT_PREDICTION = "Max. Humidity [%] (period t)"
    AVG_HUMIDITY_PERCENT_PREDICTION = "Avg. Humidity [%] (period t)"
    MIN_HUMIDITY_PERCENT_PREDICTION = "Min. Humidity [%] (period t)"
    MAX_DEW_POINT_F_PREDICTION = "Max. Dew Point [F] (period t)"
    AVG_DEW_POINT_F_PREDICTION = "Avg. Dew Point [F] (period t)"
    MIN_DEW_POINT_F_PREDICTION = "Min. Dew Point [F] (period t)"
    MAX_PRESSURE_HG_PREDICTION = "Max. Pressure [Hg] (period t)"
    AVG_PRESSURE_HG_PREDICTION = "Avg. Pressure [Hg] (period t)"
    MIN_PRESSURE_HG_PREDICTION = "Min. Pressure [Hg] (period t)"
    TOTAL_PRECIPITATION_INTENSITY_PREDICTION = (
        "Total Precipitation Intensity (period t)"
    )
    MAX_TEMP_F_HISTORY = "Max. Temperature [F] (periods [T-L, t-1])"
    AVG_TEMP_F_HISTORY = "Avg. Temperature [F] (periods [T-L, t-1])"
    MIN_TEMP_F_HISTORY = "Min. Temperature [F] (periods [T-L, t-1])"
    MAX_WIND_SPEED_MPH_HISTORY = "Max. Wind Speed [mph] (periods [T-L, t-1])"
    AVG_WIND_SPEED_MPH_HISTORY = "Avg. Wind Speed [mph] (periods [T-L, t-1])"
    MIN_WIND_SPEED_MPH_HISTORY = "Min. Wind Speed [mph] (periods [T-L, t-1])"
    MAX_HUMIDITY_PERCENT_HISTORY = "Max. Humidity [%] (periods [T-L, t-1])"
    AVG_HUMIDITY_PERCENT_HISTORY = "Avg. Humidity [%] (periods [T-L, t-1])"
    MIN_HUMIDITY_PERCENT_HISTORY = "Min. Humidity [%] (periods [T-L, t-1])"
    MAX_DEW_POINT_F_HISTORY = "Max. Dew Point [F] (periods [T-L, t-1])"
    AVG_DEW_POINT_F_HISTORY = "Avg. Dew Point [F] (periods [T-L, t-1])"
    MIN_DEW_POINT_F_HISTORY = "Min. Dew Point [F] (periods [T-L, t-1])"
    MAX_PRESSURE_HG_HISTORY = "Max. Pressure [Hg] (periods [T-L, t-1])"
    AVG_PRESSURE_HG_HISTORY = "Avg. Pressure [Hg] (periods [T-L, t-1])"
    MIN_PRESSURE_HG_HISTORY = "Min. Pressure [Hg] (periods [T-L, t-1])"
    TOTAL_PRECIPITATION_INTENSITY_HISTORY = (
        "Total Precipitation Intensity (periods [T-L, t-1])"
    )


class Feature_wo_Time(Enum):
    AMBULANCE_DEMAND = "Historic Ambulance Demand"
    HOURS = "Hour"
    WEEKDAYS = "Weekday"
    MONTHS = "Month"
    SCHOOL_HOLIDAYS = "School Holidays"
    PUBLIC_HOLIDAYS = "Public Holidays"
    EVENTS = "Events"
    MAX_TEMP_F = "Max. Temperature [F]"
    AVG_TEMP_F = "Avg. Temperature [F]"
    MIN_TEMP_F = "Min. Temperature [F]"
    MAX_WIND_SPEED_MPH = "Max. Wind Speed [mph]"
    AVG_WIND_SPEED_MPH = "Avg. Wind Speed [mph]"
    MIN_WIND_SPEED_MPH = "Min. Wind Speed [mph]"
    MAX_HUMIDITY_PERCENT = "Max. Humidity [%]"
    AVG_HUMIDITY_PERCENT = "Avg. Humidity [%]"
    MIN_HUMIDITY_PERCENT = "Min. Humidity [%]"
    MAX_DEW_POINT_F = "Max. Dew Point [F]"
    AVG_DEW_POINT_F = "Avg. Dew Point [F]"
    MIN_DEW_POINT_F = "Min. Dew Point [F]"
    MAX_PRESSURE_HG = "Max. Pressure [Hg]"
    AVG_PRESSURE_HG = "Avg. Pressure [Hg]"
    MIN_PRESSURE_HG = "Min. Pressure [Hg]"
    TOTAL_PRECIPITATION_INTENSITY = "Total Precipitation Intensity"


class FeatureCategory(Enum):
    WEATHER = "Weather"
    HOLIDAY = "Holiday"
    TIME = "Time"
    EVENTS = "Events"


class TimeCategory(Enum):
    PREDICTION = "Prediction"
    HISTORY = "History"


def assemble_feature_w_time(
    feature_wo_time: Feature_wo_Time, time_category: TimeCategory
) -> Feature_w_Time:
    return Feature_w_Time[feature_wo_time.name + "_" + time_category.name]
