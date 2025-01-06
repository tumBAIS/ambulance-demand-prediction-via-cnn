import random
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


# pylint: disable=redefined-variable-type
def adapt_inputs(
    model: Model,
    X_train: Union[np.ndarray, List[np.ndarray]],
    ninstances_train: int,
    X_test: Union[np.ndarray, List[np.ndarray]],
    ninstances_explain: int,
    multiplier: int = 1
    # transform_to_single_input: bool,
) -> Tuple[Model, np.ndarray, np.ndarray]:
    """Adapts model, test and training instances
    1. If model output is not one-dimensional, inserts flattening layer
    2. Selects correct number of training and test instances
    3. Corrects model and input data in the case of multiple inputs (e.g. for CNN)
        - Transforms input data to one-dimensional array,
        - Add slicing and reshaping layers to enable model to cope with transformed input data

    Parameters:
    model: Model
        Model to use for calculating SHAP values
    X_train: Union[np.ndarray, List[np.ndarray]]
        Data with which model has been trained
    ninstances_train: int
        Nr. of training instances that should be selected for calculating shap values
    X_test: Union[np.ndarray, List[np.ndarray]]
        Data with which model has been tested
    ninstances_explain: int
        Nr. of test instances that should be selected for calculating shap values
    multiplier: int
        Nr. of squares in data set (to adapt data set for iterative approaches)

    Returns:

    """
    # check if output is a vector or a single value, if not, insert Flatten Layer
    if (hasattr(model, "output") and hasattr(model.output, "shape")) and len(
        model.output.shape
    ) >= 3:
        model = __flatten_output_layer(model)

    X_train_instances = __select_instances(X_train, ninstances_train, True, multiplier)
    X_test_instances = __select_instances(X_test, ninstances_explain, True, multiplier)

    if isinstance(X_train_instances, list):
        model = __slice_inputs(model, X_train_instances)
        X_train_instances = __flatten_array_per_sample(X_train_instances)
        X_test_instances = __flatten_array_per_sample(X_test_instances)

    if type(X_test_instances) != type(X_train_instances):
        raise ValueError("Test and Training Instances should be of same type")

    return model, X_train_instances, X_test_instances


def flatten_nested_list(nested_lists: List[Any]) -> List[Any]:
    """Flattens a nested list ([[0,1],[2,3]]-> [0,1,2,3])

    Parameters:
    nested_list: List[Any]
        List to be flattened

    Returns:
    flattened_list: List[Any]
        Flattened list
    """
    flattened_list = []
    for nested_list in nested_lists:
        if isinstance(nested_list, list):
            flattened_list.extend(flatten_nested_list(nested_list))
        else:
            flattened_list.append(nested_list)
    return flattened_list


def __flatten_array(l: Union[List[Any], np.ndarray]) -> np.ndarray:
    """Flattens any array or list (e.g. shape(20,12,6) -> shape(1440,))

    Parameters:
    l: Union[List[Any], np.ndarray]
        List or array to be flattened

    Returns:
    np.ndarray
        Falttened array (e.g. shape(20,12,6) -> shape(1440,))
    """

    arr_flattened = np.array([])
    if isinstance(l, list):
        size = len(l)
    elif isinstance(l, np.ndarray):
        size = l.shape[0]
    else:
        return np.append(arr_flattened, np.array(l))
    for i in range(size):
        arr_flattened = np.append(arr_flattened, l[i])
    return arr_flattened


# pylint: disable=redefined-variable-type
def __flatten_array_per_sample(x: List[np.ndarray]) -> np.ndarray:
    """Creates flattened array containing all input data per sample
    (e.g. [shape(100,20,12,6), shape(100,1,1,6)] -> shape(100,1446))
    Data of all features per sample is 1) flattened 2) concatenated

    Parameters:
    l: List[Any]
        List of arrays with input data. Each array contains data of a feature
        e.g. [shape(100,20,12,6), shape(100,1,1,6), ...]

    Returns:
    np.ndarray
        Flattened array containing all input data per sample
        (e.g. [shape(100,20,12,6), shape(100,1,1,6)] -> shape(100,1446))
    """
    flattened_array = []
    nr_samples = x[0].shape[0]
    for sample in range(nr_samples):
        sample_arr = np.array([])

        # flatten tensor for each sample (e.g. shape (20,12,6) -> shape(1440,))
        nr_features = len(x)
        for feature_idx in range(nr_features):
            flattened_array_per_feature = __flatten_array(x[feature_idx][sample])
            sample_arr = np.append(sample_arr, flattened_array_per_feature)

        flattened_array.append(sample_arr)

    # shape(#samples, #input data elements)
    flattened_array = np.array(flattened_array)
    return flattened_array


def __unflatten_arr(
    X: List[np.ndarray], flattened_input: tf.Tensor
) -> List[layers.Reshape]:
    """Receives flattened tensor (flattened X). Slices and reshapes tensor such that outputs corrspond
    to original inputs (X)

    Parameters:
    X: List[np.ndarray]
        Original input data (unflattened)
    flattened_input: tf.Tensor
        Falttened X as tensor

    Returns:
    List[layers.Reshape]
        List of reshape layers. The output shape of each reshape layer corresponds to the
        original input shape of X
    """

    # shapes must be based on deepcopy as X changes during slicing
    shapes = [
        input.shape[1:] if len(input.shape) > 1 else (1,) for input in deepcopy(X)
    ]

    # slicing one-dimensional array in slices with correct nr. of elements
    slicing_layers = [
        layers.Lambda(
            lambda x: tf.slice(
                x,
                begin=[0, int(sum(np.prod(shape) for shape in shapes[0:idx]))],
                size=[-1, int(np.prod(input_shape))],
            )
        )(flattened_input)
        for idx, input_shape in enumerate(shapes)
    ]

    # reshape sliced one-dimensional arrays such that input shapes correspond to original shapes
    reshaping_layers = [
        layers.Reshape(shape, name=f"reshape_lambda_{idx}")(slicing_layers[idx])
        for (idx, shape) in enumerate(shapes)
    ]

    return reshaping_layers


def __get_output_layer_idx(
    layer: layers.Layer,
    layer_index: int,
    model: Model,
    updated_outputs: Dict[int, List[tf.Tensor]],
) -> Optional[int]:
    for layer_idx, layer_obj in enumerate(model.layers):
        for inbound_node in layer_obj.inbound_nodes:
            if (
                isinstance(inbound_node.input_tensors, list)
                and "input" not in layer_obj.name
            ) and (
                layer.output in inbound_node.input_tensors
                or
                # one of the original outputs corresponds to at leas one inbound node tensors
                len(
                    set(inbound_node.input_tensors).intersection(
                        updated_outputs[layer_index]
                    )
                )
                > 0
            ):
                return layer_idx
            if (
                inbound_node.input_tensors == layer.output
                or inbound_node.input_tensors in updated_outputs[layer_index]
            ) and "input" not in layer_obj.name:
                return layer_idx
    return None


def __add_updated_output(
    updated_outputs: Dict[int, tf.Tensor], layer_idx: int, layer: layers.Layer
):
    for inbound_node in layer.inbound_nodes:
        if (
            inbound_node.output_tensors not in updated_outputs[layer_idx]
        ):  # only add once
            updated_outputs[layer_idx].append(inbound_node.output_tensors)
        return updated_outputs


def __save_original_inbound_nodes(model: Model) -> Dict[layers.Layer, List[Any]]:
    original_inbound_nodes: Dict[layers.Layer, List[Any]] = defaultdict(list)
    for layer in model.layers:
        # original_inbound_nodes[layer] = []
        for inbound_node in layer.inbound_nodes:
            original_inbound_nodes[layer].append(inbound_node)
    return original_inbound_nodes


def __init_updated_inbound_nodes(
    model: Model,
) -> Dict[layers.Layer, List[List[Any]]]:
    original_inbound_nodes: Dict[layers.Layer, List[Any]] = {}
    for layer in model.layers:
        original_inbound_nodes[layer] = [[layer.name], []]
        for inbound_node in layer.inbound_nodes:
            original_inbound_nodes[layer][1].append(inbound_node)
    return original_inbound_nodes


def __save_original_outputs(model: Model) -> Dict[int, List[tf.Tensor]]:
    original_outputs: Dict[int, List[Any]] = defaultdict(list)
    for layer_idx, layer in enumerate(model.layers):

        # Layer is not shared among different layers
        if len(layer.outbound_nodes) == 1:
            original_outputs[layer_idx].append(layer.output)
        else:
            # Layer is shared among different layers (e.g. relu activation layer)
            # save all output tensors in original_outputs
            name_parts = layer.output.name.split("/")
            for idx_node in range(len(layer.outbound_nodes)):
                if idx_node == 0:
                    tensor_name = layer.output.name
                else:
                    tensor_name = f"{name_parts[0]}_{idx_node}/{name_parts[1]}"
                tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                    tensor_name
                )
                original_outputs[layer_idx].append(tensor)
    return original_outputs


def __replace_input_layer(
    model: Model,
    output_layer_idx: int,
    idx: int,
    sliced_inputs: List[tf.Tensor],
    updated_outputs: Dict[int, List[tf.Tensor]],
    old_name: Optional[str],
    updated_inbound_nodes: Dict[layers.Layer, List[List[Any]]],
) -> Tuple[Model, int, str, Dict[int, List[tf.Tensor]]]:

    if isinstance(model.layers[output_layer_idx].input, list):
        new_input = []
        for inbound_layer in (
            model.layers[output_layer_idx].inbound_nodes[-1].inbound_layers
        ):
            if "input" not in inbound_layer.name:
                for inbound_node in inbound_layer.inbound_nodes:
                    if (
                        # new node
                        inbound_node.output_tensors
                        not in model.layers[output_layer_idx].input
                        and
                        # don't add twice
                        inbound_node.output_tensors not in new_input
                    ):
                        new_input.append(inbound_node.output_tensors)
            else:
                new_input.append(sliced_inputs[idx])
                idx += 1

        updated_outputs = __add_updated_output(
            updated_outputs, output_layer_idx, model.layers[output_layer_idx]
        )
        old_name = model.layers[output_layer_idx].output.name
        layer_name_before_update = old_name
        del model.layers[output_layer_idx].inbound_nodes[0]
        model.layers[output_layer_idx] = model.layers[output_layer_idx](new_input)

    else:
        new_input = sliced_inputs[idx]
        old_name = model.layers[output_layer_idx].output.name
        layer_name_before_update = old_name
        del model.layers[output_layer_idx].inbound_nodes[0]
        model.layers[output_layer_idx] = model.layers[output_layer_idx](new_input)
        idx += 1

    __update_inbound_nodes(
        updated_inbound_nodes,
        model.layers[output_layer_idx],
        layer_name_before_update,
        new_input,
    )

    return model, idx, old_name, updated_outputs


# pylint: disable=too-many-nested-blocks
def __replace_following_layers(
    output_layer_output_layer_idx: Optional[int],
    model: Model,
    output_layer_idx: int,
    updated_outputs: Dict[int, tf.Tensor],
    original_inbound_nodes: Dict[layers.Layer, List[Any]],
    updated_inbound_nodes: Dict[layers.Layer, List[List[Any]]],
):
    while (
        output_layer_output_layer_idx is not None
        and "input" not in model.layers[output_layer_output_layer_idx].name
        and __all_inputs_updated(
            model.layers[output_layer_output_layer_idx],
            original_inbound_nodes,
            updated_inbound_nodes,
        )
    ):

        updated_layer = model.layers[output_layer_idx]
        layer_to_update = model.layers[output_layer_output_layer_idx]

        # save original outputs before adaptation
        updated_outputs = __add_updated_output(
            updated_outputs, output_layer_idx, updated_layer
        )
        updated_outputs = __add_updated_output(
            updated_outputs, output_layer_output_layer_idx, layer_to_update
        )

        # layer is shared by multiple layers (-> multiple inbound and outbound nodes)
        if len(layer_to_update.inbound_nodes) > 1:

            inbound_node_input_tensors = []
            inbound_node_layer = []
            # determine which input is currentyl updated
            node_idx_to_update = len(updated_inbound_nodes[layer_to_update][1]) - len(
                layer_to_update.inbound_nodes
            )
            # already updated inputs
            for idx in range(node_idx_to_update):
                inbound_node_input_tensors.append(
                    layer_to_update.inbound_nodes[idx].input_tensors
                )
                inbound_node_layer.append(
                    layer_to_update.inbound_nodes[idx].inbound_layers
                )

            # new updated input
            inbound_node_input_tensors.append(
                updated_layer.inbound_nodes[-1].output_tensors
            )
            inputs = inbound_node_input_tensors[
                -1
            ]  # save for updating updated_inbound_nodes
            inbound_node_layer.append(updated_layer)

            # remaining inputs
            for idx in range(
                node_idx_to_update + 1, len(layer_to_update.inbound_nodes)
            ):
                inbound_node_input_tensors.append(
                    layer_to_update.inbound_nodes[idx].input_tensors
                )
                inbound_node_layer.append(
                    layer_to_update.inbound_nodes[idx].inbound_layers
                )

            # delete all input and output nodes before update to ensure correct indexing
            for layer in inbound_node_layer:
                del layer_to_update.inbound_nodes[0]
                del layer.outbound_nodes[0]

            layer_name_before_update = layer_to_update.name

            # make update
            for inbound_node_input_tensor in inbound_node_input_tensors:
                model.layers[output_layer_output_layer_idx] = model.layers[
                    output_layer_output_layer_idx
                ](inbound_node_input_tensor)

        # layer is not shared by multiple layers (-> single inbound and outbound node)
        else:
            if isinstance(layer_to_update.inbound_nodes[-1].input_tensors, list):
                inputs = []
                for inbound_layer in layer_to_update.inbound_nodes[-1].inbound_layers:
                    for inbound_node in inbound_layer.inbound_nodes:
                        if (
                            # new node
                            inbound_node.output_tensors not in layer_to_update.input
                            and
                            # don't add twice
                            inbound_node.output_tensors not in inputs
                        ):
                            inputs.append(inbound_node.output_tensors)

            else:
                inputs = updated_layer.inbound_nodes[-1].output_tensors

            layer_name_before_update = layer_to_update.name

            # delete all input and output nodes before update to ensure correct indexing
            del layer_to_update.inbound_nodes[0]
            del updated_layer.outbound_nodes[0]

            # make update
            layer_to_update = layer_to_update(inputs)

        __update_inbound_nodes(
            updated_inbound_nodes, layer_to_update, layer_name_before_update, inputs
        )

        # update layer index and get following layer to update
        output_layer_idx = output_layer_output_layer_idx
        output_layer_output_layer_idx = __get_output_layer_idx(
            model.layers[output_layer_idx], output_layer_idx, model, updated_outputs
        )


def __update_inbound_nodes(
    updated_inbound_nodes: Dict[layers.Layer, List[List[Any]]],
    layer_to_update: layers.Layer,
    layer_name_before_update: Optional[str],
    inputs: tf.Tensor,
):
    if layer_to_update in updated_inbound_nodes.keys():
        updated_inbound_nodes[layer_to_update][1].append(inputs)

    else:
        for layer in updated_inbound_nodes.keys():
            if layer_name_before_update in updated_inbound_nodes[layer][0]:
                updated_inbound_nodes[layer][0].append(layer_to_update.name)
                updated_inbound_nodes[layer][1].append(inputs)

    return updated_inbound_nodes


def __slice_inputs(model: Model, X) -> Model:

    flattened_X = __flatten_array_per_sample(X)
    flattened_input = layers.Input(shape=(len(flattened_X[1])))

    sliced_inputs = __unflatten_arr(X, flattened_input)

    idx = 0
    updated_outputs = __save_original_outputs(model)
    original_inbound_nodes = __save_original_inbound_nodes(model)
    updated_inbound_nodes = __init_updated_inbound_nodes(model)
    old_name = None

    # tracks how many inputs have been updated if multiple inputs are given
    # assumption: inputs are updated in the given order

    for layer_idx, layer in enumerate(model.layers):

        if "input" in layer.name:

            output_layer_idx = __get_output_layer_idx(
                layer, layer_idx, model, updated_outputs
            )

            if output_layer_idx is not None:

                output_layer_output_layer_idx = __get_output_layer_idx(
                    model.layers[output_layer_idx],
                    output_layer_idx,
                    model,
                    updated_outputs,
                )

                model, idx, old_name, updated_outputs = __replace_input_layer(
                    model,
                    output_layer_idx,
                    idx,
                    sliced_inputs,
                    updated_outputs,
                    old_name,
                    updated_inbound_nodes,
                )

                __replace_following_layers(
                    output_layer_output_layer_idx,
                    model,
                    output_layer_idx,
                    updated_outputs,
                    original_inbound_nodes,
                    updated_inbound_nodes,
                )

            if idx == len(sliced_inputs):
                break

    new_model = Model(inputs=flattened_input, outputs=model.layers[-1].output)

    return new_model


def __all_inputs_updated(
    layer_to_check: layers.Layer,
    original_inbound_nodes: Dict[layers.Layer, List[Any]],
    updated_inbound_nodes: Dict[layers.Layer, List[List[Any]]],
) -> bool:

    for inbound_node in layer_to_check.inbound_nodes:
        if isinstance(inbound_node.inbound_layers, list):
            inbound_layers = inbound_node.inbound_layers
        else:
            inbound_layers = [inbound_node.inbound_layers]

        for inbound_layer in inbound_layers:
            if len(original_inbound_nodes[inbound_layer]) * 2 != len(
                updated_inbound_nodes[inbound_layer][1]
            ):
                return False

            # layer may have been renamed, check alternative names
            for layer in updated_inbound_nodes.keys():
                if inbound_layer in updated_inbound_nodes[layer][0] and len(
                    original_inbound_nodes[inbound_layer]
                ) * 2 != len(updated_inbound_nodes[layer][1]):
                    return False

        return True


def __flatten_output_layer(model: Model) -> Model:
    """Inserts Flatten Layer if output is not a vector or a single value

    Parameters
    model: keras.models.Model
        Model instance without an output vector or single value

    Returns
    model: keras.models.Model
        Model instance including Flatten output layer
    """
    fl_layer = layers.Flatten()(model.layers[-1].output)
    model2 = Model(inputs=model.input, outputs=[fl_layer])
    model = model2
    return model


def __select_instances(
    X: Union[None, List[Any], np.ndarray],
    n: int,
    draw_instances_randomly: bool = True,
    multiplier: int = 1,
) -> list:
    """Select n instances from X

    Parameters:
    X: list
        data
    n: int
        number of instances to select
    multiplier: int
        Number of samples multipler, such that same instances are compares for CNN and iterative approaches.
        For iterative approaches, this should be equal to rows * columns
    """

    if n is None:
        return X

    if draw_instances_randomly:
        random.seed(0)
        if isinstance(X, list):
            random_indices = random.sample(range(len(X[0])), n)
            random_indices = __get_random_indices_for_iterative_approaches(
                random_indices, multiplier
            )
            X_new = []
            for x in X:
                X_new.append(np.array([x[i] for i in random_indices]))

        else:
            random_indices = random.sample(range(int(X.shape[0] / multiplier)), n)
            random_indices = __get_random_indices_for_iterative_approaches(
                random_indices, multiplier
            )
            X_new = [X[i] for i in random_indices]
            X_new = np.array(X_new)

    else:
        if isinstance(X, list):
            X_new = []
            for x in X:
                X_new.append(x[0:n])
        else:
            X_new = X[0:n]

    return X_new


def __get_random_indices_for_iterative_approaches(random_indices, multiplier):
    return [(i * multiplier) + m for i in random_indices for m in range(multiplier)]
