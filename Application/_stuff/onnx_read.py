import onnx
#import numpy as np

if __name__ == "__main__":
    # Load the ONNX model
    model = onnx.load("./NNVct/Application/_stuff/PT_test_model.onnx")

    # Print the model's graph (nodes, inputs, and outputs)
    print("Model Inputs:")
    for input in model.graph.input:
        print(input.name)

    print("\nModel Outputs:")
    for output in model.graph.output:
        print(output.name)

    print("\nModel Nodes (Layers/Operations):")
    for node in model.graph.node:
        print(node.op_type, node.name)

    print("\n\n\n\n\n")

    for initializer in model.graph.initializer:
        # The initializer name
        print("Initializer: ", initializer.name)
        
        # Convert the initializer to a numpy array
        # Initializers are stored in protobuf format, so they need to be converted
        np_tensor = onnx.numpy_helper.to_array(initializer)[:5]
        
        print(np_tensor.shape)
        print(np_tensor, end="\n\n")
        
