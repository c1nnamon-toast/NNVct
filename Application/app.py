from flask import Flask, render_template, jsonify, request
import numpy as np
import time
import json
import os

#from extract_NN1 import ultimate_knowledge as model_data # temporary

app = Flask(__name__)


@app.route('/abstract_layout')
def abstract_layout():
    return render_template('abstract_layout_page.html')


@app.route('/layout', methods=['GET', 'POST'])
def layout():
    if request.method == 'POST':
        start_time = time.time();

        with open('C:/Users/darks/Documents/VNV/NNVct/Application/model_info.json', 'r') as json_file:
            model_data = json.load(json_file)
        # the new code should be here

        # Extract weights between first two layers
        weights_first_to_second = model_data['Weights'][0][0] # Ignore biases for now
        num_red_nodes = model_data['Layers'][0][2]
        num_orange_nodes = model_data['Layers'][0][3]

        #print(num_red_nodes, num_orange_nodes, weights_first_to_second)

        # Assign colors based on whether the original opacity was negative
        #colors = np.where(random_opacities < 0, 'red', 'green')

        # Create edges using NumPy arrays for colors and opacities

        edges = [
            {
                'data': {
                    'id': f'edge{i}-{j}',
                    'source': f'redNode{i}',
                    'target': f'orangeNode{j}',
                    'color': 'green' if float(weights_first_to_second[j][i]) > 0  else 'red',
                    'opacity': float(weights_first_to_second[j][i]),
                }
            }
            for i in range(num_red_nodes) for j in range(num_orange_nodes)
        ]

        response_json = jsonify({
            'numRedNodes': num_red_nodes,
            'numOrangeNodes': num_orange_nodes,
            'edges': edges
        })

        
        print(f"Backend processing took {time.time() - start_time} seconds.")

        return response_json

    return render_template('layout_page.html')

if __name__ == '__main__':
    app.run(debug=True)