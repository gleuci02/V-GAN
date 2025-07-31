import os
import numpy as np
from sklearn.preprocessing import StandardScaler
#from skimage.io import imread
#from skimage.transform import resize

def load_coil20(data_dir="data/coil-20/", image_size=(32, 32)):
   
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory {data_dir} does not exist.")

    # Initialize lists for features and labels
    X, y = [], []

    # Load each image and its label
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".png"):
            label = int(file_name.split("_")[0][3:])  # Extract label from filename
            img_path = os.path.join(data_dir, file_name)
            
            # Read and preprocess the image
            image = imread(img_path, as_gray=True)  # Load image as grayscale
            image_resized = resize(image, image_size, anti_aliasing=True)  # Resize image
            X.append(image_resized.flatten())  # Flatten into a 1D vector
            y.append(label)

    # Convert to numpy arrays and scale features
    X = np.array(X)
    X = StandardScaler().fit_transform(X)  # Standardize features
    y = np.array(y)

    return X, y