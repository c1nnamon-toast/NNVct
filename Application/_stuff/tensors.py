import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(1, 4)  
        self.fc2 = nn.Linear(4, 8)  
        self.fc3 = nn.Linear(8, 2)  
        self.fc4 = nn.Linear(2, 1)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  
        return x


model = SimpleMLP()

for layer in model.children():
    layer_weights = list(layer.parameters());
    print(layer_weights[0], layer_weights[1], sep='\n', end='\n\n\n')
