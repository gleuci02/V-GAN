import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import random


def search_json_and_return(json_obj, target_string, key_name=None):
    matching_values = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, (dict, list)):
                matching_values.extend(search_json_and_return(
                    value, target_string, key_name=key))
            elif isinstance(value, str) and target_string in value:
                matching_values.append({key: value})
    elif isinstance(json_obj, list):
        for item in json_obj:
            if isinstance(item, (dict, list)):
                matching_values.extend(search_json_and_return(
                    item, target_string, key_name=key_name))
            elif isinstance(item, str) and target_string in item:
                matching_values.append([key_name, item])
    if matching_values.__len__() > 2:
        print(
            f"Warning: {matching_values.__len__()} matching datasets found: {matching_values}.")
        return None
    return matching_values


def load_dataset_path(dataset_name) -> Path:
    try:
        with open("./data/datasets_files_name.json", "r") as json_file:
            json_obj = json.load(json_file)
            matching_values = search_json_and_return(json_obj, dataset_name)
            if matching_values is not None and not matching_values.__len__() == 0:
                print(f"Loading dataset: {matching_values[0]}")
                dataset = matching_values[0]
                return Path("data/datasets/" + dataset[0] + "/" + dataset[1]).absolute()
            elif matching_values is None:
                print(f"Specify the dataset name more precisely.")
            else:
                print(f"Dataset not found.")
    except FileNotFoundError:
        print(f"Json file not found in the root directory. Please download the file using the instructions in the README.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON. File might be corrupted.")


def load_data(dataset_name):
    np.random.seed(777)
    random.seed(777)
    os.environ["PYTHONHASSEED"] = str(777)

    data = np.load(load_dataset_path(dataset_name))
    df = pd.DataFrame(data["X"])
    df["outlier"] = data["y"]
    df["id"] = df.index

    df["outlier"] = pd.factorize(df["outlier"], sort=True)[
        0]  # Keep in mind: 0 inlier, 1 outlier

    inlier = df[df["outlier"] == 0]

    train = inlier.sample(frac=0.8)
    test = df.drop(train.index)

    train_x = train.drop(axis=1, labels=["outlier", "id"])
    test_x = test.drop(axis=1, labels=["outlier", "id"])
    test_y = pd.DataFrame(columns=["y"])
    test_y["y"] = test["outlier"]

    return normalize(train_x.to_numpy()), normalize(test_x.to_numpy()), test_y.to_numpy()


if __name__ == '__main__':
    """Test if the datasets have been downloaded correctly
    """
    datasets = load_dataset_path("cover")
    print(datasets)
    if os.path.isfile(datasets):
        print("Datasets have been dowloaded correctly")
    else:
        print("Error downloading datasets")
