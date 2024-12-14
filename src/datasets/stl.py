from torchvision import datasets
from sklearn.preprocessing import StandardScaler


def load_stl10():
    stl10 = datasets.STL10('./data/stl', split='train', download=True)
    print(stl10.data.shape)
    X = StandardScaler().fit_transform(stl10.data.reshape(5000, 3 * 96 * 96))
    y = stl10.labels
    return X, y