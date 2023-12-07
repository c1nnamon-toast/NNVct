import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, topology):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(topology) - 1):
            self.layers.append(nn.Linear(topology[i], topology[i + 1]))
            
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


if __name__ == "__main__":
    data = pd.read_csv('C:/Users/darks/Documents/python/python_old/NeuralNetworks/NN/skuska/Concrete_Data.csv')

    X = data.iloc[:, :-1] 
    y = data.iloc[:, -1]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1)

    model = MLP([8, 32, 128, 64, 32, 1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 500
    best_val_loss = np.inf
    epochs_no_improve = 0
    n_epochs_stop = 25

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()  # training mode
        optimizer.zero_grad()   
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()  # evaluation mode
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print('Early stopping!')
                break

    # best model
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    # for x in model.children():
    #     print(len(list(x.parameters())[0]), len(list(x.parameters())))
    #     break;

    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')


    plt.figure(figsize=(10, 8))
    plt.plot(range(epoch+1), train_losses, label='Training Loss')
    plt.plot(range(epoch+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()