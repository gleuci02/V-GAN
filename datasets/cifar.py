from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler


def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    cifar10_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transform)
    #X = StandardScaler().fit_transform(cifar10.data.reshape(50000, 32*32*3).data)
    #y = cifar10.targets
    #return X, y
    return cifar10_train, cifar10_test

def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    cifar100_train = datasets.CIFAR100('./data/cifar', train=True, download=True, transform=transform)
    cifar100_test = datasets.CIFAR100('./data/cifar', train=False, download=True, transform=transform)
    #X = StandardScaler().fit_transform(cifar100.data.reshape(50000, 32*32*3).data)
    #y = cifar100.targets
    #return X, y
    return cifar100_train, cifar100_test