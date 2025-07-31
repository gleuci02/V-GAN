from torchvision import datasets
from sklearn.preprocessing import StandardScaler


def load_cifar10():
    cifar10 = datasets.CIFAR10('./data/cifar', train=True, download=True)
    X = StandardScaler().fit_transform(cifar10.data.reshape(50000, 32*32*3).data)
    y = cifar10.targets
    return X, y

def load_cifar100():
    cifar100 = datasets.CIFAR100('./data/cifar', train=True, download=True)
    X = StandardScaler().fit_transform(cifar100.data.reshape(50000, 32*32*3).data)
    y = cifar100.targets
    return X, y