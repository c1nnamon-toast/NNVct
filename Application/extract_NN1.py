import torch
#import numpy as np
from simple_NN import SimpleMLP as Network # ????
#import inspect
import json
import requests

if __name__ == "__main__":
    model = Network();
    model.load_state_dict(torch.load('./NNVct/Application/model.pth'))

    ultimate_knowledge = dict();
    ultimate_knowledge['Layers'] = [];
    ultimate_knowledge['Activation functions'] = [('fc1', 'ReLU'), ('fc2', 'ReLU'), ('fc3', 'ReLU')]; # change in future
    ultimate_knowledge['Weights'] = [];     

    for name, module in model.named_modules():
        class_name = type(module).__name__ 
        ultimate_knowledge['Layers'].append((name, class_name));

    for layer in model.children():
        layer_weights = list(layer.parameters());
        ultimate_knowledge['Weights'].append(layer_weights);

    json_data = json.dumps(ultimate_knowledge)

    url = "http://localhost:5000/hell"  # Replace with your Flask endpoint URL
    response = requests.post(url, json=json_data)