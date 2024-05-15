import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 4, bias=False) 
        self.fc2 = nn.Linear(4, 6, bias=False) 
        self.fc3 = nn.Linear(6, 14, bias=False)  
        self.fc4 = nn.Linear(14, 8, bias=False) 
        self.fc5 = nn.Linear(8, 4, bias=False) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  
        return x

if __name__ == "__main__":
    model = SimpleMLP()
    model.eval()
    sample_input = torch.randn(1, 2)
    torch.onnx.export(model, sample_input, './NNVct/Application/backend/model_PT.onnx', opset_version=11)