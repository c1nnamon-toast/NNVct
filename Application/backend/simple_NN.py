import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 4) 
        self.fc2 = nn.Linear(4, 6) 
        self.fc3 = nn.Linear(6, 14)  
        self.fc4 = nn.Linear(14, 8) 
        self.fc5 = nn.Linear(8, 4) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  
        return x


if __name__ == "__main__":
    model = SimpleMLP()

    torch.save(model.state_dict(), './NNVct/Application/backend/model.pth')