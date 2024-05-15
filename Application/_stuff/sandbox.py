import onnx
import numpy as np

# Load the ONNX model
model_path = './NNVct/Application/backend/model_TF.onnx'
onnx_model = onnx.load(model_path)

# Function to print tensor information
def print_tensor_info(tensor):
    # We use numpy to convert the raw data of the tensor into a numpy array
    np_array = np.frombuffer(tensor.raw_data, dtype=np.float32)
    # Reshape the array to the dimensions of the tensor
    np_array = np_array.reshape(tensor.dims)
    print("Shape:", np_array.shape)

# Iterate through each initializer (weights/biases)
for initializer in onnx_model.graph.initializer:
    print(f"{initializer.name}:")
    print_tensor_info(initializer)