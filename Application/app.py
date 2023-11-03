from flask import Flask, render_template, jsonify, request
import numpy as np
import time

app = Flask(__name__)


@app.route('/abstract_layout')
def abstract_layout():
    return render_template('abstract_layout_page.html')


@app.route('/layout', methods=['GET', 'POST'])
def layout():
    if request.method == 'POST':
        start_time = time.time();


        num_red_nodes = 20
        num_orange_nodes = int(request.form.get('numOrangeNodes', 1))
        num_edges = num_red_nodes * num_orange_nodes

        # Generate random opacities between -1 and 1
        random_opacities = 2 * (np.random.rand(num_edges) - 0.5)

        # Convert to absolute values for opacity
        opacities = np.abs(random_opacities)

        # Assign colors based on whether the original opacity was negative
        colors = np.where(random_opacities < 0, 'red', 'green')

        # Create edges using NumPy arrays for colors and opacities
        edges = [
            {
                'data': {
                    'id': f'edge{i}-{j}',
                    'source': f'redNode{i}',
                    'target': f'orangeNode{j}',
                    'color': colors[i * num_orange_nodes + j],
                    'opacity': float(opacities[i * num_orange_nodes + j])  # Cast to float for JSON serialization
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