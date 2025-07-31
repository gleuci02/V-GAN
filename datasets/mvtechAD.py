from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler


def load_vials():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)), #128  96  126
        #transforms.Lambda(lambda x: x.view(-1))
    ])
    vials_train = datasets.ImageFolder(root='/mnt/simhomes/leuci/V-GAN/data/vials/vial/train', transform=transform)
    vials_test = datasets.ImageFolder(root='/mnt/simhomes/leuci/V-GAN/data/vials/vial/train', transform=transform)
    
    return vials_train, vials_test

def load_fruit_jelly():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        #transforms.Lambda(lambda x: x.view(-1))
    ])
    fruit_jelly_train = datasets.ImageFolder(root='/mnt/simhomes/leuci/V-GAN/data/fruitJelly/fruit_jelly/train', transform=transform)
    fruit_jelly_test = datasets.ImageFolder(root='/mnt/simhomes/leuci/V-GAN/data/fruitJelly/fruit_jelly/train', transform=transform)
    
    return fruit_jelly_train, fruit_jelly_test