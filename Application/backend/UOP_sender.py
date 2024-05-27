import re
import onnx
from onnx import numpy_helper

from universal_onnx_parser import extract_model_info

def extract_activation_names(layers):
    activation_names = []

    for layer in layers:
        # The activation function name is the second element in the tuple
        activation = layer[1]

        if activation:
            # Split the activation function name by '/' and get the last part
            activation_parts = activation.split('/')
            activation_name = activation_parts[-1] if activation_parts else activation

            # Remove the underscore and the number at the end of the activation function name
            activation_name = re.sub(r'_\d+$', '', activation_name)
            activation_name = activation_name.lower()
            activation_names.append(activation_name)
        else:
            activation_names.append('')

    return activation_names


def extract_layer_names(layers):
    layer_names = []

    for layer in layers:
        # The layer name is the first element in the tuple
        layer_name = layer[0]

        # Use a regular expression to get the part of the layer name from the end until the second '/' from the end
        match = re.search(r'([^/]+)/[^/]+$', layer_name)
        if match:
            layer_name = match.group(1)

        layer_names.append(layer_name)

    return layer_names


def extract_weights(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    weights = []

    # Iterate over all the initializers (weights) in the model
    for initializer in model.graph.initializer:
        # Convert the initializer to a numpy array
        weight = numpy_helper.to_array(initializer)
        weights.append(weight)
        print(weight.shape)

    return weights


if __name__ == "__main__":
    # TF
    path = "./NNVct/Application/backend/model_TF.onnx"
    layers = extract_model_info(path)
    layers_names = extract_layer_names(layers)
    activation_names = extract_activation_names(layers)
    weights = extract_weights(path)

    print(layers_names, activation_names, [w.shape for w in weights], sep='\n', end='\n\n')

    # PT
    path = "./NNVct/Application/backend/model_PT.onnx"
    layers = extract_model_info(path)
    layers_names = extract_layer_names(layers)
    activation_names = extract_activation_names(layers)
    weights = extract_weights(path)
    
    print(layers_names, activation_names, [w.shape for w in weights], sep='\n', end='\n\n')