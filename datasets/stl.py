from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler


def load_stl10():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # Convert to PyTorch tensor with shape [1, 28, 28]
        #transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])
    stl10_train = datasets.STL10('./data/stl', split='train', download=True, transform=transform)
    stl10_test = datasets.STL10('./data/stl', split='train', download=False, transform=transform)
    #X = StandardScaler().fit_transform(stl10.data.reshape(5000, 3 * 96 * 96))
    #y = stl10.labels
    #return X, y
    return stl10_train, stl10_test
