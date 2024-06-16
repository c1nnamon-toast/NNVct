import json
from backend.UOP_loadinfo import extract_model_info
from backend.universal_onnx_parser import extract_layers_info, extract_weights


def format_layer_info(layer_name, activation_name, weight):
    """
    Format the layer information into the desired structure, including the activation function directly within the layer info.

    Parameters:
    - layer_name (str): The name of the layer.
    - activation_name (str): The name of the activation function.
    - weight (numpy.ndarray): The weight matrix of the layer.

    Returns:
    - Tuple[List, List]: A tuple containing the formatted layer information and weight information as lists.
    """
    layer_info = [layer_name, "Linear", weight.shape[1], weight.shape[0], activation_name]
    weight_info = weight.tolist()


    return layer_info, weight_info


def save_model_to_json(model_path, json_filename):
    """
    Saves the onnx model to a customly formated JSON file.

    Parameters:
    - model_path (str): The file path to the ONNX model.
    - json_filename (str): The file path for saving the JSON output.
    """
    # Extract the model information
    layers = extract_model_info(model_path)
    layer_and_activation_info = extract_layers_info(layers)
    weights = extract_weights(model_path)

    # Format the model information
    layers_info = []
    weights_info = []
    for (layer_name, activation_name), weight in zip(layer_and_activation_info, weights):
        layer_info, weight_info = format_layer_info(layer_name, activation_name, weight)
        layers_info.append(layer_info)
        weights_info.append(weight_info)
        # print(len(weight_info))

    # Prepare the data for the JSON file
    data = {
        "Layers": layers_info,
        "Weights": weights_info
    }

    # Save the data to a JSON file
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)






# Testing
if __name__ == "__main__":
    save_model_to_json("./NNVct/Application/backend/model.onnx", "./NNVct/Application/backend/model.json")