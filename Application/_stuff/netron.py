import torch
from simple_NN import SimpleMLP

model = SimpleMLP()
dummy_input = torch.randn(1, 8)  # Example input
torch.onnx.export(model, dummy_input, "model.onnx")