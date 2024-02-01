import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(12, 25) 
        self.fc2 = nn.Linear(15, 20) 
        self.fc3 = nn.Linear(20, 30)  
        self.fc4 = nn.Linear(30, 15) 
        self.fc5 = nn.Linear(15, 7) 
        self.fc6 = nn.Linear(7, 3) 
        self.fc7 = nn.Linear(3, 1) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  
        return x

# Create an instance of the MLP
model = SimpleMLP()

# Print the model architecture
#print(model)
torch.save(model.state_dict(), './NNVct/Application/backend/model.pth')