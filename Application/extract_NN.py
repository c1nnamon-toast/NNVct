from testNN0 import MLP
from torch import save

if __name__ == "__main__":

    topology = [8, 32, 128, 64, 32, 1]
    model = MLP(topology)

    print(model);

    save(model, './NNVct/Application/model.pth')
