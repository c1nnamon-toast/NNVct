import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    torch_model = MyModel()
    torch_input = torch.randn(1, 1, 32, 32)
    path = "./NNVct/Application/_stuff/PT_test_model.onnx"

    # With weights 
    onnx_program = torch.onnx.export(torch_model, 
                                     torch_input,
                                     path,
                                     verbose=True)

    # Without weights 
    # onnx_program = torch.onnx.export(torch_model, 
    #                                  torch_input,
    #                                  path,
    #                                  export_params=False,
    #                                  verbose=True)

    # In "training" state 
    # onnx_program = torch.onnx.export(torch_model, 
    #                                  torch_input,
    #                                  path,
    #                                  training=torch.onnx.TrainingMode.TRAINING) ??? TrainingMode.PRESERVE ???
