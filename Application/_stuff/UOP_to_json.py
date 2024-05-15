import json
from UOP_sender import extract_model_info, extract_layer_names, extract_activation_names, extract_weights

def format_layer_info(layer_name, activation_name, weight):
    # Format the layer information into the desired structure
    layer_info = [layer_name, "Linear", weight.shape[1], weight.shape[0]]
    activation_info = [layer_name, activation_name]
    weight_info = weight.tolist()

    return layer_info, activation_info, weight_info

def save_model_info_to_json(model_path, json_filename):
    # Extract the model information
    layers = extract_model_info(model_path)
    layer_names = extract_layer_names(layers)
    activation_names = extract_activation_names(layers)
    weights = extract_weights(model_path)

    # Format the model information
    layers_info = []
    activations_info = []
    weights_info = []
    for layer_name, activation_name, weight in zip(layer_names, activation_names, weights):
        layer_info, activation_info, weight_info = format_layer_info(layer_name, activation_name, weight)
        layers_info.append(layer_info)
        activations_info.append(activation_info)
        weights_info.append(weight_info)

    # Prepare the data for the JSON file
    data = {
        "Layers": layers_info,
        "Activation functions": activations_info,
        "Weights": weights_info
    }

    # Save the data to a JSON file
    with open(json_filename, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    save_model_info_to_json("./NNVct/Application/backend/model_TF.onnx", "./NNVct/Application/backend/model_TF.json")
    save_model_info_to_json("./NNVct/Application/backend/model_PT.onnx", "./NNVct/Application/backend/model_PT.json")