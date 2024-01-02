import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 6)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(6, 8)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(8, 3)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(3, 1)  # Third hidden layer to fourth hidden layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation function for the output layer
        return x

# Create an instance of the MLP
model = SimpleMLP()

# Print the model architecture
#print(model)
torch.save(model.state_dict(), './NNVct/Application/model.pth')