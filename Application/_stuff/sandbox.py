import json

with open('C:/Users/darks/Documents/VNV/NNVct/Application/model_info.json', 'r') as json_file:
    model_data = json.load(json_file)

weights = model_data['Weights'][0];
print(weights);