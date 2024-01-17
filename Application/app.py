from flask import Flask, render_template, jsonify, request
import numpy as np
import time
import json
import os
import random


app = Flask(__name__)


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



@app.route('/abstractLayout')
def abstract_layout():
    layers = 5
    nodes = [{"data": {"id": f"layer{i}", "label": f"Layer {i}"}} for i in range(layers)]
    edges = [{"data": {"source": f"layer{i}", "target": f"layer{i+1}"}} for i in range(layers - 1)]
    return render_template('abstract_layout_page.html', elements={"nodes": nodes, "edges": edges})



@app.route('/layout', methods=['GET', 'POST'])
def layout():
    if request.method == 'POST':
        start_time = time.time()

        dimensions = request.json
        containerWidth = dimensions['width']
        containerHeight = dimensions['height']

        with open('C:/Users/darks/Documents/VNV/NNVct/Application/backend/model_info.json', 'r') as json_file:
            model_data = json.load(json_file)

        nodes = []
        edges = []

        total_layers = len(model_data['Layers']) + 1  # Include input layer
        layerWidth = containerWidth / total_layers

        # Input layer nodes
        input_features = model_data['Layers'][0][2]  # Number of input features from the first layer
        nodeHeight = containerHeight / input_features
        for j in range(input_features):
            node_id = f'input_{j}'
            nodes.append({
                'group': 'nodes',
                'grabbable': False, 
                'position': {
                    'x': layerWidth / 2,
                    'y': nodeHeight * j + nodeHeight / 2
                },
                'data': {
                    'id': node_id,
                    'layer': 'input',
                    'type': 'Input'
                }
            })

        # Iterate through non-zero layers to create nodes
        for i, layer_info in enumerate(model_data['Layers']):
            layer_name, layer_type, in_features, out_features = layer_info

            for j in range(out_features):
                node_id = f'{layer_name}_{j}'
                nodeHeight = containerHeight / out_features

                nodes.append({
                    'group': 'nodes',
                    'grabbable': False, 
                    'position': {
                        'x': layerWidth * (i + 1) + layerWidth / 2,
                        'y': nodeHeight * j + nodeHeight / 2
                    },
                    'data': {
                        'id': node_id,
                        'layer': layer_name,
                        'type': layer_type
                    }
                })


        # Input layer edges
        fc1_output_features = model_data['Layers'][0][3]  # Number of output features for fc1

        for input_node_index in range(input_features):
            for fc1_node_index in range(fc1_output_features):

                edge_id = f'edge_input_{input_node_index}-to-fc1_{fc1_node_index}'
                source_id = f'input_{input_node_index}'
                target_id = f'fc1_{fc1_node_index}'

                weight = random.uniform(-1, 1)
                edge_color = 'green' if weight > 0 else 'red'
                opacity = (weight + 1) / 2  # Normalize weight to the range [0, 1]

                edges.append({
                    'group': 'edges',
                    'data': {
                        'id': edge_id,
                        'source': source_id,
                        'target': target_id,
                        'lineColor': edge_color,
                        'opacity': opacity
                    }
                })

        # Edges between non-zero layers
        for i in range(len(model_data['Layers']) - 1):
            source_layer_info = model_data['Layers'][i]
            target_layer_info = model_data['Layers'][i + 1]
            for source_node_index in range(source_layer_info[3]):
                for target_node_index in range(target_layer_info[3]):
                    edge_id = f'edge_{source_layer_info[0]}_{source_node_index}-to-{target_layer_info[0]}_{target_node_index}'
                    source_id = f'{source_layer_info[0]}_{source_node_index}'
                    target_id = f'{target_layer_info[0]}_{target_node_index}'

                    weight = random.uniform(-1, 1)
                    opacity = (weight + 1) / 2  # Normalize weight to the range [0, 1]
                    edge_color = 'green' if weight > 0 else 'red'

                    edges.append({
                    'group': 'edges',
                    'data': {
                        'id': edge_id,
                        'source': source_id,
                        'target': target_id,
                        #'weight': weight,
                        'lineColor': edge_color,
                        'opacity': opacity  # Add opacity to the edge data
                    }
                    })

        response_json = jsonify({
            'nodes': nodes,
            'edges': edges
        })

        print(f"Backend processing took {time.time() - start_time} seconds.")
        return response_json

    return render_template('layout_page.html')




if __name__ == '__main__':
    app.run(debug=True)