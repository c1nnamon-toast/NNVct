import onnx
import re

def extract_model_info(model_path):
    """
    Extracts names of layers and their corresponding activation functions based on the provided ONNX model. *Doesn't extract weights*

    Reads an ONNX model from the given file path and extracts a list of layer names paired with their associated activation functions, if any. 
    The activation functions are identified based on a predefined list of activation function names stored in a text file. 

    Parameters:
    - model_path (str): The file path to the ONNX model.

    Returns:
    - List[Tuple[str, str]]: A list of tuples, where each tuple contains:
        - The name of the layer.
        - The name of the activation function associated with the layer; an empty string if none.
    """
    """
    Notes:
    - The function assumes that activation functions are specified as the last part of the node name in the model's graph.
    - The function handles potential naming conventions where activation functions might have a suffix indicating
      their index (e.g., "_1"). Pytorch (PT) models are known to have such naming conventions.
    - The activation function names are expected to be in lowercase in the 'activation_functions.txt' file.
    - Weight extraction is avoided due to the complexity of the output structure and the potential size of the weights.
    """
    model = onnx.load(model_path)

    # List with tuples of size 2, where the first element is the layer name and the second element is the activation function
    layers = []

    # File of considered activation functions, (to lower case)  
    with open('./NNVct/Application/backend/activation_functions.txt', 'r') as f:
        activations = [line.strip().lower() for line in f]

    previous_layer = None
    previous_activation = ""
    for i, node in enumerate(model.graph.node):
        # The activiations are at the end of the node name
        # Split the node name by '/' and get the last part (to lower case)
        node_name_parts = node.name.lower().split('/')
        node_name_last_part = node_name_parts[-1] if node_name_parts else node.name.lower()

        # PT adds an underscore and a number at the end of the activation function name
        # Regex to ignore both the underscore and the number
        node_name_last_part = re.sub(r'_\d+$', '', node_name_last_part)

        # Check what we are currently looking at activation or a true layer
        # If the TL doesn't have an activation function, i.e. TL comes after a TL
        # then the previous TL doesn't have an activation function
        if node_name_last_part in activations:
            previous_activation = node.name
        else:
            if previous_layer:
                layers.append((previous_layer, previous_activation))
            previous_layer = node.name
            previous_activation = ""

    # Last layer check
    if previous_layer:
        layers.append((previous_layer, previous_activation))

    return layers




# Testing

# TF usage example
def TF():
    layers = extract_model_info("./NNVct/Application/backend/model_TF.onnx")

    # Fancy printing
    print(f"Model Information for ./NNVct/Application/backend/model_TF.onnx:\n")
    print("Layers:")
    for i, layer in enumerate(layers):
        print(f"Layer {i + 1}:   {layer[0]},   {layer[1] if layer[1] else 'None'}")

# PT usage example
def PT():
    layers = extract_model_info("./NNVct/Application/backend/model_PT.onnx")

    # Fancy printing
    print(f"Model Information for ./NNVct/Application/backend/model_PT.onnx:\n")
    print("Layers:")
    for i, layer in enumerate(layers):
        print(f"Layer {i + 1}:   {layer[0]},   {layer[1] if layer[1] else 'None'}")

if __name__ == "__main__":
    TF()
    # PT()
