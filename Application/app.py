import time
import json
from flask import Flask, render_template, jsonify, request

from backend.render_NN import loadfullNN, loadNNpartially, getLayers, process_node

PATH = "./Application/backend/model_TF.json"

app = Flask(__name__)
app.secret_key = 'dripus'

@app.route('/abstractLayout')
def abstract_layout():
    with open(PATH, 'r') as json_file:
        model_data = json.load(json_file)

    #layers = 150
    layers = len(model_data['Layers']) + 1
    layersInfo = getLayers(PATH)
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

    model_info_path = PATH
    
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

        model_info_path = PATH
        
        response = loadfullNN(model_info_path, containerWidth, containerHeight)
        response_json = jsonify(response)

        #print(response)
        print(f"Backend processing took {time.time() - start_time} seconds.")
        return response_json

    return render_template('layout.html')



@app.route('/api/activation/<node_id>')
def api_activation(node_id):
    if "input" in node_id.lower():
        return jsonify(proceed=False, message="Stay on current page")
    else:
        model_info_path = PATH
        weights, activation_function = process_node(model_info_path, node_id)
        return jsonify(proceed=True, weights=weights, activation_function=activation_function, node_id=node_id)

@app.route('/visualizeActivation/<node_id>')
def visualize_activation(node_id):

    return render_template('activation_visualization.html', node_id=node_id)



if __name__ == '__main__':
    app.run(debug=True)