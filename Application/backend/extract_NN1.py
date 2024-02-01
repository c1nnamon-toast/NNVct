import torch
#import numpy as np
from simple_NN import SimpleMLP as Network # ????
#import inspect
import json
import requests

if __name__ == "__main__":
    model = Network();
    model.load_state_dict(torch.load('./NNVct/Application/backend/model.pth'))

    ultimate_knowledge = dict();
    ultimate_knowledge['Layers'] = [];
    ultimate_knowledge['Activation functions'] = [('fc1', 'ReLU'), ('fc2', 'ReLU'), ('fc3', 'ReLU'), ('fc4', 'ReLU'), ('fc5', 'ReLU'), ('fc6', 'ReLU')]; # change in future
    ultimate_knowledge['Weights'] = [];     

    for name, module in model.named_modules():
        class_name = type(module).__name__ 
        if(class_name == 'Linear'):
            ultimate_knowledge['Layers'].append((name, class_name, module.in_features, module.out_features));

    for i, layer in enumerate(model.children()):
        list_of_tensors = list(layer.parameters());
        #print(list_of_tensors)
        layer_weights = [t.detach().tolist() for t in list_of_tensors];
        ultimate_knowledge['Weights'].append(layer_weights)

        #ultimate_knowledge['Weights'].append(layer_weights);

    #print(ultimate_knowledge);

    with open('./NNVct/Application/backend/model_info.json', 'w') as json_file:
        json.dump(ultimate_knowledge, json_file)