from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

def convert_to_rgb(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def load_caltech101():
    transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        #transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    caltech101_train = datasets.Caltech101('./data/caltech', download=True, transform=transform)
    caltech101_test = datasets.Caltech101('./data/caltech', download=True, transform=transform)
    return caltech101_train, caltech101_test

def load_caltech256():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        #transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    caltech256_train = datasets.Caltech256('./data/caltech', train=True, download=True, transform=transform)
    caltech256_test = datasets.Caltech256('./data/caltech', train=False, download=True, transform=transform)
    return caltech256_train, caltech256_test