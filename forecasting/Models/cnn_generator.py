import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from Data_Handler.data_storage import DataStorage, generate_training_test_data
from Data_Handler.feature_names import (
    FEATURE_SETTINGS_LABEL_CONCATENATION,
    FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME,
    FEATURE_SETTINGS_LABEL_INPUT_SHAPE)
from Models.helper import (get_activation, get_loss_function, get_metrics,
                           get_optimizer)
from Models.types import DataType, ModelType
from Results_Handler.results_handler import BestModel, output_results
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Concatenate, Conv3D, Conv3DTranspose,
                                     Dense, Dropout, Input, LocallyConnected2D,
                                     Reshape)
from tensorflow.keras.models import Model

FIXED_SETTINGS_FILE = "./Settings/CNN/settings_cnn_fixed.yaml"

# pylint: disable=no-member, too-many-instance-attributes
class CNN:
    def __init__(self, **kwargs):

        # Set statically defined hyperparameters via yaml file
        self.id = str(datetime.today()).replace(":", "-")

        with open(FIXED_SETTINGS_FILE, "r") as stream:
            try:
                fixed_settings = yaml.safe_load(stream)
                for key, value in fixed_settings.items():
                    setattr(self, key, value)
            except yaml.YAMLError as exc:
                print(exc)

        # Dynamically defined hyperparameters via Settings/Bayesian Optimization (settings_cnn.csv)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # overwrite string values read from settings file by corresponding objects
        self.metrics = get_metrics(*self.metrics)
        self.optimizer = get_optimizer(self.optimizer, self.learning_rate)
        self.loss_function = get_loss_function(self.loss_function_name)
        self.activation_3dconv_1 = get_activation(self.activation_3dconv_1)
        self.activation_3dconv_2 = get_activation(self.activation_3dconv_2)
        self.activation_3dconv_3 = get_activation(self.activation_3dconv_3)
        self.activation_3d_transp_conv = get_activation(self.activation_3d_transp_conv)
        self.activation_local_con = get_activation(self.activation_local_con)
        self.activation_dense_1 = get_activation(self.activation_dense_1)
        self.activation_dense_2 = get_activation(self.activation_dense_2)

        # dictionary with all cnn attributes for printing results
        dic_attributes = self.__dict__.copy()
        del dic_attributes["model"]
        self.data = list(dic_attributes.values())
        self.data_headers = list(dic_attributes.keys())

    def to_concatenate_1d(self, features_df: pd.DataFrame):
        concatenation = []
        features_names = features_df.index[
            tuple([features_df[FEATURE_SETTINGS_LABEL_CONCATENATION] == "1D"])
        ].tolist()

        for f in features_names:
            if isinstance(features_df.loc[f][FEATURE_SETTINGS_LABEL_INPUT_SHAPE], int):
                sample_shape = features_df.loc[f][FEATURE_SETTINGS_LABEL_INPUT_SHAPE]
            else:
                sample_shape = tuple(
                    int(i)
                    for i in features_df.loc[f][
                        FEATURE_SETTINGS_LABEL_INPUT_SHAPE
                    ].split(",")
                )
            inputs = Input(shape=sample_shape)
            concatenation.append(inputs)
        return concatenation

    # pylint: disable=too-many-locals
    def build_cnn_model(
        self, features_df: pd.DataFrame, total_look_back: int, nrRows: int, nrCols: int
    ) -> Model:

        features_df2 = features_df.sort_values(
            by=[
                FEATURE_SETTINGS_LABEL_CONCATENATION,
                FEATURE_SETTINGS_LABEL_FEATURE_WO_TIME_NAME,
            ],
            ascending=[False, True],
        )

        to_concat_1d = features_df2[
            features_df2[FEATURE_SETTINGS_LABEL_CONCATENATION] == "1D"
        ]  # features concatenated after locally connected layer (1D)
        to_concat_2d = features_df2[
            features_df2[FEATURE_SETTINGS_LABEL_CONCATENATION] == "2D"
        ]  # features concatenated in 2D
        to_concat_3d = features_df2[
            features_df2[FEATURE_SETTINGS_LABEL_CONCATENATION] == "3D"
        ]  # features concatenated in 3D (upsampled via transposed convolution)

        # Input historical call data (cube)
        sample_shape = (nrRows, nrCols, total_look_back, 1)
        inputs = Input(shape=sample_shape)

        # first and second 3d conv layer with activation
        conv3d_layer_1 = Conv3D(
            self.nrFilters_3dconv_1,
            kernel_size=(
                self.kernalSize_3dconv_1,
                self.kernalSize_3dconv_1,
                self.kernalSize_3dconv_1,
            ),
            padding="same",
            kernel_regularizer=regularizers.l2(self.l2_kernel_regularizer_3dconv_1),
        )(inputs)

        if self.activation_3dconv_1 is not None:
            conv3d_layer_1 = self.activation_3dconv_1(conv3d_layer_1)

        conv3d_layer_2 = Conv3D(
            self.nrFilters_3dconv_2,
            kernel_size=(
                self.kernalSize_3dconv_2,
                self.kernalSize_3dconv_2,
                self.kernalSize_3dconv_2,
            ),
            padding="same",
            kernel_regularizer=regularizers.l2(self.l2_kernel_regularizer_3dconv_2),
        )(conv3d_layer_1)

        if self.activation_3dconv_2 is not None:
            conv3d_layer_2 = self.activation_3dconv_2(conv3d_layer_2)

        if to_concat_3d.shape[0] > 0:
            inputs_concat3d = []
            layers_to_concat = []
            shapes = [
                tuple(map(int, i.split(",")))
                for i in to_concat_3d[FEATURE_SETTINGS_LABEL_INPUT_SHAPE]
            ]

            for shape in shapes:

                # define inputs to be upsampled
                sample_shape = (1, 1, total_look_back, 1)
                inputs_3d = Input(shape=sample_shape)
                inputs_concat3d.append(inputs_3d)

                # check if input data is correct
                if shape != sample_shape:
                    raise ValueError(
                        "Shape of inputs upsampled by transposed conv. must be of shape ",
                        sample_shape,
                        " but are of shape ",
                        shapes[0],
                    )

                # transposed conv layer with activation
                conv3d_trans_layer_1 = Conv3DTranspose(
                    self.nrFilters_3d_transp_conv,
                    (nrRows, nrCols, 1),
                    kernel_regularizer=regularizers.l2(
                        self.l2_kernel_regularizer_3d_transp_conv
                    ),
                )(inputs_3d)
                if self.activation_3d_transp_conv is not None:
                    conv3d_trans_layer_1 = self.activation_3d_transp_conv(
                        conv3d_trans_layer_1
                    )
                layers_to_concat.append(conv3d_trans_layer_1)

            # concatenate data upsampled by transposed conv layer
            concatenated_3d_layer = Concatenate()([conv3d_layer_2, *layers_to_concat])
            conv3d_layer_2 = concatenated_3d_layer

        # temporal fusion
        conv3d_layer_3 = Conv3D(
            self.nrFilters_3dconv_3,
            kernel_size=(1, 1, total_look_back),
            kernel_regularizer=regularizers.l2(self.l2_kernel_regularizer_3dconv_3),
        )(conv3d_layer_2)
        if self.activation_3dconv_3 is not None:
            conv3d_layer_3 = self.activation_3dconv_3(conv3d_layer_3)
        temporal_fused_layer = Reshape((nrRows, nrCols, self.nrFilters_3dconv_3))(
            conv3d_layer_3
        )

        # concatenate 2d data
        if to_concat_2d.shape[0] > 0:
            sample_shape = (nrRows, nrCols, 1)
            inputs_concat2d = Input(shape=sample_shape)
            temporal_fused_layer = Concatenate()(
                [temporal_fused_layer, inputs_concat2d]
            )

        # locally connected to 1d array
        outputs = LocallyConnected2D(1, kernel_size=(1, 1))(temporal_fused_layer)
        if self.activation_local_con is not None:
            outputs = self.activation_local_con(outputs)
        outputs = Reshape((nrRows * nrCols,))(outputs)

        # concatenate 1d data
        if to_concat_1d.shape[0] > 0:
            inputs_concat1d = self.to_concatenate_1d(features_df2)
            outputs = Concatenate()([outputs, *inputs_concat1d])

        # dense layers
        d1 = Dense(units=self.nrUnits_dense_1)(outputs)
        if self.activation_dense_1 is not None:
            d1 = self.activation_dense_1(d1)
        dropout1 = Dropout(self.dropout_dense_1)(d1)

        d2 = Dense(units=nrRows * nrCols)(dropout1)
        if self.activation_dense_2 is not None:
            d2 = self.activation_dense_2(d2)
        outputs = Reshape((nrRows, nrCols))(d2)

        # generate inputs
        inputs = [inputs]
        if to_concat_3d.shape[0] > 0:
            inputs.extend(inputs_concat3d)
        if to_concat_2d.shape[0] > 0:
            inputs.extend([inputs_concat2d])
        if to_concat_1d.shape[0] > 0:
            inputs.extend(inputs_concat1d)

        model = Model(inputs=inputs, outputs=outputs, name="CNN")

        self.model = model

        return model


# pylint: disable=too-many-arguments, too-many-locals
def run_CNN(
    settings: Dict[str, Any],
    data_storage: DataStorage,
    total_look_back: int,
    incident_types: List[str],
    nrRows: int,
    nrCols: int,
    best_model: Optional[BestModel] = None,
    X: Optional[List[np.ndarray]] = None,
    y: Optional[np.ndarray] = None,
    X_train: Optional[List[np.ndarray]] = None,
    y_train: Optional[np.ndarray] = None,
    X_test: Optional[List[np.ndarray]] = None,
    y_test: Optional[np.ndarray] = None,
    output_path: str = "./Results",
):

    cnn = CNN(**settings)
    modeltype = ModelType.CNN
    datatype = DataType.LAYER_BASED

    # only features marked as included and filtered by filter array included in features_df
    features_df = data_storage.get_filtered_features_df()
    features_dict = data_storage.get_features_dict(datatype, True)

    if X_train is None or y_train is None or X_test is None or y_test is None:

        # get data (if necessary)
        if (
            (X is None or y is None)
            and (datatype in data_storage.X.keys())
            and (datatype in data_storage.y.keys())
        ):
            X = data_storage.X[datatype]  # already filtered
            y = data_storage.y[datatype]  # already filtered
        else:
            raise ValueError("Please provide X or y or store data in data_storage.")

        X_train, y_train, X_test, y_test = generate_training_test_data(
            *X,
            y=y,
            test_size=cnn.test_size,
            random_state=cnn.random_state,
            shuffle_data=cnn.shuffle_data,
        )

    # Generate model
    model = cnn.build_cnn_model(
        features_df, total_look_back=total_look_back, nrRows=nrRows, nrCols=nrCols
    )

    if cnn.early_stopping:
        callback = EarlyStopping(
            monitor=cnn.early_stopping_monitor,
            patience=cnn.early_stopping_patience,
            restore_best_weights=cnn.restore_best_weights,
        )

    model.summary()

    # tf.keras.utils.plot_model(model)

    # Compile and fit model
    model.compile(loss=cnn.loss_function, optimizer=cnn.optimizer, metrics=cnn.metrics)

    start = time.time()
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=cnn.batch_size,
        epochs=cnn.no_epochs,
        verbose=cnn.verbosity,
        callbacks=[callback],
        validation_split=cnn.split_validation,
    )
    end = time.time()
    runtime = (end - start) / 60  # in minutes

    # Evaluate results
    score = model.evaluate(X_test, y_test, verbose=0)

    # Write out results
    output_results(
        output_path=output_path,
        modeltype=modeltype,
        model=cnn,
        features=features_dict,
        runtime=runtime,
        incident_types=incident_types,
        history=history,
        score=score,
    )

    # Documents data (model, training and test data, feature configurations)
    best_model.update_best_model_info(
        score[1], settings, model, X_train, y_train, X_test, y_test, features_dict
    )

    return score[1]
