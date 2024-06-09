import re
import onnx
from onnx import numpy_helper
from backend.UOP_loadinfo import extract_model_info


def extract_layers_info(layers):
    """
    Extracts names of layers and their corresponding activation functions.
    """
    layer_and_activation_info = []
    for layer in layers:
        layer_name, activation = layer

        # Extract a simplified layer name
        match = re.search(r'([^/]+)/[^/]+$', layer_name)
        simple_layer_name = match.group(1) if match else layer_name

        # Normalize the activation function name
        if activation:
            activation_parts = activation.split('/')
            activation_name = activation_parts[-1] if activation_parts else activation
            activation_name = re.sub(r'_\d+$', '', activation_name).lower()
        else:
            activation_name = ''

        layer_and_activation_info.append((simple_layer_name, activation_name))


    return layer_and_activation_info


def extract_weights(model_path):
    """
    Loads an ONNX model and extracts all the weight tensors as numpy arrays.    
    """
    model = onnx.load(model_path)
    weights = []
    for initializer in model.graph.initializer:
        weight = numpy_helper.to_array(initializer)
        weights.append(weight)
        # print(weight.shape)


    return weights





# Testing

if __name__ == "__main__":
    path = "./NNVct/Application/backend/model_TF.onnx"
    layers = extract_model_info(path)
    layer_and_activation_info = extract_layers_info(layers)
    weights = extract_weights(path)


    print([info[0] for info in layer_and_activation_info], [info[1] for info in layer_and_activation_info], [w.shape for w in weights], sep='\n', end='\n\n')


    # path = "./NNVct/Application/backend/model_PT.onnx"
    # layers = extract_model_info(path)
    # layer_and_activation_info = extract_layers_info(layers)
    # weights = extract_weights(path)
    
    # print([info[0] for info in layer_and_activation_info], [info[1] for info in layer_and_activation_info], [w.shape for w in weights], sep='\n', end='\n\n')