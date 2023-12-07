import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 30)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(30, 70)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(70, 20)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(20, 10)  # Third hidden layer to fourth hidden layer

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