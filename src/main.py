from datasets.mnist import load_mnist, load_fashion_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
from vgan import VGAN
from pathlib import Path
import datetime
from torch.utils.data import DataLoader

DATASETS = {
    "CIFAR10": load_cifar10,
    "CIFAR100": load_cifar100,
    "MNIST": load_mnist,
    "FASHION_MNIST": load_fashion_mnist,
    "STL10": load_stl10,
}

if __name__ == "__main__":
    dataset = DATASETS["MNIST"]()
    model = VGAN(epochs = 1500, temperature=10, batch_size=60000, path_to_directory=Path()/ "experiments" / f"Example_dataset_{datetime.datetime.now()}", iternum_d=1, iternum_g=5,lr_G = 0.01, lr_D = 0.01)
    dataloader = DataLoader(dataset, batch_size=60000, shuffle=False)
    print(enumerate(dataloader))
    for batch_idx, (X, y) in enumerate(dataloader):
        model.fit(X)
        u = model.generate_subspaces(500)
        print(u)