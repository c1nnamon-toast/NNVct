import json
import os
import random

def loadNNpartially(model_path, containerWidth, containerHeight, layers=5):
    with open(model_path, 'r') as json_file:
        model_data = json.load(json_file)
    
    nodes = []
    edges = []

    total_layers = len(model_data['Layers']) + 1  # Include input layer
    layerWidth = containerWidth / total_layers
    max_nodes_on_screen = 13
    max_possible_nodes = max( model_data['Layers'][0][3], max([x[3] for x in model_data['Layers']]) )
    
    phi = containerHeight/max_nodes_on_screen

    print(max_possible_nodes, phi)

    # Input layer nodes
    input_features = model_data['Layers'][0][2]  # Number of input features from the first layer
    nodeHeight = containerHeight / input_features
    for j in range(input_features):

        print(phi*(max_possible_nodes - input_features)/2 + phi*j)

        node_id = f'input_{j}'
        nodes.append({
            'group': 'nodes',
            'grabbable': False, 
            'position': {
                'x': int(layerWidth / 2),
                'y': int(phi*(max_possible_nodes - input_features)/2 + phi*j)
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

            print(phi*(max_possible_nodes - out_features)/2 + phi*j)

            nodes.append({
                'group': 'nodes',
                'grabbable': False, 
                'position': {
                    'x': int(layerWidth * (i + 1) + layerWidth / 2),
                    'y': int(phi*(max_possible_nodes - out_features)/2 + phi*j)
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

    return {'nodes': nodes, 'edges': edges}



def loadfullNN(model_path, containerWidth, containerHeight):
    with open(model_path, 'r') as json_file:
        model_data = json.load(json_file)
    
    nodes = []
    edges = []

    total_layers = len(model_data['Layers']) + 1  # Include input layer
    layerWidth = containerWidth / total_layers
    max_nodes_on_screen = 13
    max_possible_nodes = max( model_data['Layers'][0][3], max([x[3] for x in model_data['Layers']]) )
    
    phi = containerHeight/max_nodes_on_screen

    print(max_possible_nodes, phi)

    # Input layer nodes
    input_features = model_data['Layers'][0][2]  # Number of input features from the first layer
    nodeHeight = containerHeight / input_features
    for j in range(input_features):

        print(phi*(max_possible_nodes - input_features)/2 + phi*j)

        node_id = f'input_{j}'
        nodes.append({
            'group': 'nodes',
            'grabbable': False, 
            'position': {
                'x': int(layerWidth / 2),
                'y': int(phi*(max_possible_nodes - input_features)/2 + phi*j)
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

            print(phi*(max_possible_nodes - out_features)/2 + phi*j)

            nodes.append({
                'group': 'nodes',
                'grabbable': False, 
                'position': {
                    'x': int(layerWidth * (i + 1) + layerWidth / 2),
                    'y': int(phi*(max_possible_nodes - out_features)/2 + phi*j)
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

    return {'nodes': nodes, 'edges': edges}