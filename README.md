# Ambulance Demand Prediction via Convolutional Neural Networks

This software predicts ambulance demand applying a Convolutional Neural Networks and is able to benchmark against MLP, Decision Trees and Random Forests.

This method is proposed in:

> Maximiliane Rautenstrauß and Maximilian Schiffer. Ambulance Demand Prediction via Convolutional Neural Networks. arXiv preprint [arXiv:2306.04994](https://arxiv.org/abs/2306.04994).

All code components required to run the code for the instances considered in the paper are provided here. 

## Data

Due to copy right and memory restrictions, we can not provide all data sets in this repository, however, all data sets are publicly available.

We provide:
- `Public_Holidays_Seattle.csv` containing the public holidays for Seattle
- `School_Holidays_Seattle.csv` containing the school holidays for Seattle

The following data can not be provided in this repository but is publicly available:
- `Seattle_Neighborhoods.geojson` containing the polygons of Seattle's neighborhoods. 
    - Retrieved from https://www.openstreetmap.org/
- `Seattle_Real_Time_Fire_911_Calls.csv` containing the emergency calls of Seattle
    - Retrieved from https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj
- `Special_Events_Permits_Seattle.csv` containing the events for Seattle
    - Retrieved from https://data.seattle.gov/Permitting/Special-Events-Permits/dm95-f8w5
- `Weather_Seattle.csv` containing the weather data for Seattle
    - Retrieved from https://wunderground.com/history

All data instances should be saved in the directory `Data` under the corresponding name. We provide a data template defining the required data structure in the corresponding files.

## Settings

The directory `Settings` contains:
- `settings_main.py` which configures the main settings, e.g. which model to run, which data to use etc. (see comments in `settings_main.py` for example values)
- CNN
    - `settings_cnn_fixed.py` contains all parameters for the CNN that are not tuned
    - `settings_cnn.py` contains all parameters with its default values for the CNN that can be tuned
    - `settings_cnn_space.csv` defines the parameters to be tuned and the corresponding search space
- MLP
    - `settings_mlp_fixed.py` contains all parameters with its default values  for the MLP that are not tuned
    - `settings_mlp.py` contains all parameters for the MLP that can be tuned
    - `settings_mlp_space.csv` defines the parameters to be tuned and the corresponding search space
- Trees
    - `settings_dectree.py` contains all parameters with its default values for the Decision Tree that can be adapted
    - `settings_ranforest.py` contains all parameters  with its default values for the Random Forest that can be adapted
- Features
    - `Feature_Settings.csv` defines which features to include and additional characteristics (e.g. whether a feature should be one-hot-encoded)
    - `Feature_Settings_24h.csv` similar to `Feature_Settings.csv`, however, neglects feature "Hours"

## Getting started

1. Setup and activate virtual environment
```
python3 -m venv <path-to-new-virtual-environment>
source <path-to-new-virtual-environment>/bin/activate
```

2. Install dependencies

```
make install
```

3. Define required settings in `Settings` directory

4. Run model and/or tuning process
```
python <path-to-forecasting-directory>/Forecast.py --settings="<path-to-main-settings-file>"
```

For typical instance and neural network sizes, a GPU should be used.

## Remaining directories overview
- `Data`: contains all necessary data / data templates
- `Bayesian_Optimization`: contains all files needed to conduct hyperparameter tuning
- `Data_Handler`: reads, transforms and stores datasets
- `Models`: generates and runs models (cnn, mlp, decision trees, random forests, Medic method)
- `Results_Handler`: stores incumbent parameters and dumps results during tuning process
- `Shap`: calculates and visualizes Shap values

## Licenses
- The code applied for `Bayesian_Optimization` is based on [scikit-optimize](https://github.com/scikit-optimize) which is licensed  as followed: 
BSD 3-Clause License,
Copyright (c) 2016-2020 The scikit-optimize developers.
All rights reserved.
