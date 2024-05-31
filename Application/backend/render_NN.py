import json
from math import ceil 


def getLayers(model_path):
    with open(model_path, 'r') as json_file:
        model_data = json.load(json_file)
    
    nodes = []
    edges = []

    smort = [['input', '', 0, model_data['Layers'][0][2]]] # Include input layer
    smort.extend(model_data['Layers'])


    return smort



def process_node(path, node_id):
    # Split the node_id into layer and neuron index
    layer, neuron_index = node_id.rsplit('_', 1)
    neuron_index = int(neuron_index)

    with open(path, 'r') as json_file:
        model_data = json.load(json_file)

    # Find the index of the given layer
    layer_index = None
    for i, layer_data in enumerate(model_data['Layers']):
        if layer_data[0] == layer:
            layer_index = i
            break

    # Find the index of the given activation
    activation_index = None
    for i, activation_data in enumerate(model_data['Activation functions']):
        if activation_data[0] == layer:
            activation_index = i
            break

    if activation_index is None:
        return {'error': 'Activation not found'}

    neuron_weights = model_data['Weights'][layer_index][neuron_index]
    activation_function = model_data['Activation functions'][activation_index][1]


    return neuron_weights, activation_function



def loadNNpartially(model_path, containerWidth, containerHeight, centerlayer, layers=5):
    with open(model_path, 'r') as json_file:
        model_data = json.load(json_file)
    
    nodes = []
    edges = []

    smort = [['input', '', 0, model_data['Layers'][0][2]]] # Include input layer
    smort.extend(model_data['Layers'])

    left, right = max(1, centerlayer - int((layers - 1) / 2)), min(centerlayer + ceil((layers - 1) / 2), len(smort))
    smort = smort[left-1:right]
    total_layers = len(smort) 

    # print(smort, left, right)

    layerWidth = containerWidth / total_layers
    max_nodes_on_screen = 13 # something is off with calculations
    max_possible_nodes = max([x[3] for x in smort])
    phi = containerHeight/max_nodes_on_screen

    # Nodes
    create_nodes(nodes, smort, layerWidth, max_possible_nodes, phi, full=False, left=left)

    # Edges
    create_edges(edges, smort, right - left + 1, model_data, full=False, left=left)


    return {'nodes': nodes, 'edges': edges}



def loadfullNN(model_path, containerWidth, containerHeight):
    with open(model_path, 'r') as json_file:
        model_data = json.load(json_file)
    
    nodes = []
    edges = []

    smort = [['input', '', 0, model_data['Layers'][0][2]]] # Include input layer
    smort.extend(model_data['Layers'])

    total_layers = len(smort) 

    layerWidth = containerWidth / total_layers
    max_nodes_on_screen = 13 # something is off with calculations
    max_possible_nodes = max([x[3] for x in smort])
    phi = containerHeight/max_nodes_on_screen


    # Nodes
    create_nodes(nodes, smort, layerWidth, max_possible_nodes, phi, full=True)
    
    # Edges
    create_edges(edges, smort, total_layers, model_data, full=True)


    return {'nodes': nodes, 'edges': edges}



def create_nodes(nodes, smort, layerWidth, max_possible_nodes, phi, full, left=None):
    for i, layer_info in enumerate(smort):
        layer_name, layer_type, in_features, out_features = layer_info

        for j in range(out_features):
            node_id = f'{layer_name}_{j}'
            if(full):
                if(i == 0):
                    node_id = f'input_{j}'
            else:
                if((i + left) == 1):
                    node_id = f'input_{j}'
            

            nodeWidth = int(layerWidth*i + layerWidth / 2);
            nodeHeight = int(phi*(max_possible_nodes - out_features)/2 + phi*j)

            #print(phi*(max_possible_nodes - out_features)/2 + phi*j)

            nodes.append({
                'group': 'nodes',
                'grabbable': False, 
                'position': {
                    'x': nodeWidth,
                    'y': nodeHeight,
                },
                'data': {
                    'id': node_id,
                    'layer': layer_name,
                    'type': layer_type
                }
            })



def create_edges(edges, smort, total_layers, model_data, full, left=None):
    for i in range(0, total_layers - 1):
               
        source_layer_info = smort[i]
        target_layer_info = smort[i + 1]

        for source_node_index in range(source_layer_info[3]):
            for target_node_index in range(target_layer_info[3]):
                edge_id = f'edge_{source_layer_info[0]}_{source_node_index}-to-{target_layer_info[0]}_{target_node_index}'
                source_id = f'{source_layer_info[0]}_{source_node_index}'
                target_id = f'{target_layer_info[0]}_{target_node_index}'
                if(full):
                    weight = model_data['Weights'][i][target_node_index][source_node_index]
                else:
                    weight = model_data['Weights'][left + i - 1][target_node_index][source_node_index]
                # weight = random.uniform(-1, 1)

                opacity = abs(weight*0.8)  # abs for negative weights
                edge_color = 'green' if weight > 0 else 'red'
                edge_gradient = '#94dc79 #68c981 #42b588 #239f8a' if (edge_color == 'green') else '#9e00b8 #df0072 #ff0000' # green & red
                
                edges.append({
                'group': 'edges',
                'data': {
                    'id': edge_id,
                    'source': source_id,
                    'target': target_id,
                    #'weight': weight,

                    'lineGradient': edge_gradient,
                    'lineColor': edge_color,
                    'opacity': opacity  # Add opacity to the edge data
                }
                })