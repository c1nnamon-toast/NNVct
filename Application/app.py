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
    with open('C:/Users/darks/Documents/VNV/NNVct/Application/backend/model_info.json', 'r') as json_file:
        model_data = json.load(json_file)

    #layers = 150
    layers = len(model_data['Layers']) + 1
    layersInfo = getLayers('C:/Users/darks/Documents/VNV/NNVct/Application/backend/model_info.json')
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

    model_info_path = 'C:/Users/darks/Documents/VNV/NNVct/Application/backend/model_info.json'
    
    response = loadNNpartially(model_info_path, containerWidth, containerHeight, int(layernum) + 1) # 1-based indexing

    print(response)
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

        model_info_path = 'C:/Users/darks/Documents/VNV/NNVct/Application/backend/model_info.json'
        
        response = loadfullNN(model_info_path, containerWidth, containerHeight)
        response_json = jsonify(response)

        print(response)
        print(f"Backend processing took {time.time() - start_time} seconds.")
        return response_json

    return render_template('layout.html')



@app.route('/visualizeRelu/<node_id>')
def visualize_relu(node_id):
    value = request.args.get('value', type=float)  # Get the calculated value from query parameter
    if value is None:
        # If no value is provided, handle it appropriately (e.g., set a default or return an error)
        value = 0  # Example default value

    return render_template('relu_visualization.html', node_id=node_id, calculated_value=value)

@app.route('/processNode/<node_id>')
def process_node(node_id):
    # Perform your calculations here based on the node_id
    # For example, calculate a value based on the node's connected edges
    # ...

    print(node_id)

    with open('C:/Users/darks/Documents/VNV/NNVct/Application/backend/model_info.json', 'r') as json_file:
        model_data = json.load(json_file)

    weights = model_data['Weights'][0];
    print(weights);

    calculated_value = 1.6  # Replace with your calculation logic

    return jsonify({'calculatedValue': calculated_value})



if __name__ == '__main__':
    app.run(debug=True)