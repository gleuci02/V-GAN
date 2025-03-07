from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

#def load_mnist():
#    mnist = fetch_openml('mnist_784', version=1)
#    X = StandardScaler().fit_transform(mnist.data)
#    y = mnist.target
#    return X, y

def load_mnist():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        #transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    #X = StandardScaler().fit_transform(mnist.data.reshape(60000, 28*28).data)
    #y = mnist.targets
    #return X, y
    return mnist_train, mnist_test

def load_fashion_mnist():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        #transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    fashion_mnist_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    #X = StandardScaler().fit_transform(fashion_mnist.data.reshape(60000, 28*28).data)
    #y = fashion_mnist.targets
    #return X, y
    return fashion_mnist_train, fashion_mnist_test