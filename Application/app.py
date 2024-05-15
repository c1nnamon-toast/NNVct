import time
import json
import os
from flask import Flask, render_template, jsonify, request, session
import numpy as np

from backend.render_NN import loadfullNN, loadNNpartially, getLayers


app = Flask(__name__)
app.secret_key = 'dripus'

@app.route('/abstractLayout')
def abstract_layout():
    with open('C:/Users/Miguel/Documents/VNV/NNVct/Application/backend/model_info.json', 'r') as json_file:
        model_data = json.load(json_file)

    #layers = 150
    layers = len(model_data['Layers']) + 1
    layersInfo = getLayers('C:/Users/Miguel/Documents/VNV/NNVct/Application/backend/model_info.json')
    neuronnumbers = [x[3] for x in layersInfo]
    max_neurons = max(neuronnumbers);
    
    nodes = [{"data":   {
                        "id": f"layer_{i}", 
                        "label": f"Layer {i}", 
                        "width": int(max(0.2, neuronnumbers[i]/max_neurons)*100/2),
                        "height": int(max(0.2, neuronnumbers[i]/max_neurons)*100),
                        }} for i in range(layers)]
    edges = [{"data":   {
                        "source": f"layer_{i}", 
                        "target": f"layer_{i+1}"
                        }} for i in range(layers - 1)]
    
    print(nodes);

    return render_template('abstract_layout.html', elements={"nodes": nodes, "edges": edges})



@app.route('/processNodeForFocusedLayout', methods=['POST'])
def process_node_for_focused_layout():
    data = request.json
    layernum = data['selectedLayer'].split('_')[-1]
    containerWidth = data['width']
    containerHeight = data['height']

    model_info_path = 'C:/Users/Miguel/Documents/VNV/NNVct/Application/backend/model_info.json'
    
    response = loadNNpartially(model_info_path, containerWidth, containerHeight, int(layernum) + 1) # 1-based indexing

    #print(response)
    # Send the response back to the client
    return jsonify(response)



@app.route('/focusedLayout')
def focused_layout():
    return render_template('focused_layout.html')



@app.route('/layout', methods=['GET', 'POST'])
def layout():
    if request.method == 'POST':
        start_time = time.time()

        dimensions = request.json
        containerWidth = dimensions['width']
        containerHeight = dimensions['height']

        model_info_path = 'C:/Users/Miguel/Documents/VNV/NNVct/Application/backend/model_TF.json'
        
        response = loadfullNN(model_info_path, containerWidth, containerHeight)
        response_json = jsonify(response)

        #print(response)
        print(f"Backend processing took {time.time() - start_time} seconds.")
        return response_json

    return render_template('layout.html')



@app.route('/visualizeActivation/<node_id>')
def visualize_relu(node_id):
    value = request.args.get('value', type=float)  # Get the calculated value from query parameter
    if value is None:
        # If no value is provided, handle it appropriately (e.g., set a default or return an error)
        value = 0  # Example default value

    activation_function = 'ReLU'

    return render_template('activation_visualization.html', node_id=node_id, calculated_value=value)



@app.route('/processNode/<node_id>')
def process_node(node_id):
    print(node_id)

    # Split the node_id into layer and neuron index
    layer, neuron_index = node_id.rsplit('_', 1)
    neuron_index = int(neuron_index)

    with open('C:/Users/Miguel/Documents/VNV/NNVct/Application/backend/model_TF.json', 'r') as json_file:
        model_data = json.load(json_file)

    # Find the index of the given layer
    layer_index = None
    for i, layer_data in enumerate(model_data['Layers']):
        if layer_data[0] == layer:
            layer_index = i
            break
    if layer_index is None:
        return jsonify({'error': 'Layer not found'})

    # Get the weights for the given layer
    weights = model_data['Weights'][layer_index]

    # Check if neuron index is valid
    if neuron_index >= len(weights):
        return jsonify({'error': 'Neuron index out of range'})

    # Extract the weights corresponding to the neuron index
    neuron_weights = weights[neuron_index]

    print(f"Weights that come into the neuron {node_id}: {neuron_weights}")

    # Assume some input values for now
    input_values = [1.0 for _ in neuron_weights]

    # Calculate the weighted sum
    weighted_sum = sum(weight * input_value for weight, input_value in zip(neuron_weights, input_values))

    return jsonify({'weightedSum': weighted_sum})



if __name__ == '__main__':
    app.run(debug=True)