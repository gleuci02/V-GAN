from src.modules.od_module import VGAN, VMMD
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.feature_bagging import FeatureBagging
from pathlib import Path
import datetime
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score, f1_score
from sel_suod.models.base import sel_SUOD
import itertools
from sklearn.preprocessing import label_binarize
from joblib.externals.loky import get_reusable_executor
from data.get_datasets import load_data
import json
import os
import logging
from src.modules.tools import aggregator_funct
logger = logging.getLogger(__name__)


def launch_outlier_detection_experiments(dataset_name: str, base_estimators: list, epochs: int = 10,
                                         temperature: float = 10, seed: int = 777, gen_model_to_use: str = "VGAN") -> dict:
    """Launch the outlier detection experiments for a given dataset

    Args:
        dataset_name (str): Name of the dataset to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, 
        then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        tuple: Returns the AUC, PRAUC and F1 of the ensemble obtained by VGAN subspaces
    """
    logger.info(
        "No instance of a pretrained generation model found. Proceeding to train a new Generator.")
    X_train, X_test, y_test = load_data(dataset_name)

    if gen_model_to_use == "VGAN":
        vgan = VGAN(epochs=epochs, temperature=temperature, batch_size=500,
                    path_to_directory=Path() / "experiments" / "VGAN"
                    f"VGAN_{dataset_name}",
                    iternum_d=1, iternum_g=5, lr_G=0.01, lr_D=0.01)
    elif gen_model_to_use == "VMMD":
        vgan = VMMD(epochs=epochs,  batch_size=500,
                    path_to_directory=Path() / "experiments" / "VMMD" /
                    f"VMMD_{dataset_name}", lr=0.01)
    else:
        raise ValueError(f"{gen_model_to_use} is not a generator in the list")

    vgan.fit(X_train)
    vgan.seed = seed
    vgan.approx_subspace_dist(add_leftover_features=False)
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=vgan.subspaces,
                              n_jobs=-1, bps_flag=False, approx_flag_global=False)
    ensemble_model.fit(X_train)
    decision_function_scores_ens = ensemble_model.decision_function(
        X_test)
    decision_function_scores_ens = aggregator_funct(
        decision_function_scores_ens, weights=vgan.proba, type="avg")
    return {"Dataset": dataset_name,
            "AUC": auc(y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(y_test, decision_function_scores_ens),
            "F1": f1_score(y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1)}


def pretrained_launch_outlier_detection_experiments(dataset_name: str, base_estimators: list, seed: int = 777, gen_model_to_use: str = "VGAN") -> tuple:
    """Launch the outlier detection experiments for a given dataset

    Args:
        dataset_name (str): Name of the dataset to load
        base_estimators (list): List including all base estimators to build the ensemble. If the length is = 1, 
        then an homogeneus ensemble will be fitted.
        directory (Path): Path to the directory one wishes to load to
    Returns:
        np.array: Returns the AUC, PRAUC and F1
    """
    logger.info(
        f"Pretrained generator found!")
    X_train, X_test, y_test = load_data(dataset_name)

    if gen_model_to_use == "VGAN":
        logger.debug("Loading VGAN")
        vgan = VGAN()
        vgan.load_models(Path() / "experiments" / "VGAN" /
                         f"VGAN_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    elif gen_model_to_use == "VMMD":
        logger.debug("Loading VMMD")
        vgan = VMMD()
        vgan.load_models(Path() / "experiments" / "VMMD" /
                         f"VMMD_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    vgan.seed = seed
    vgan.approx_subspace_dist(add_leftover_features=False)
    ensemble_model = sel_SUOD(base_estimators=base_estimators, subspaces=vgan.subspaces,
                              n_jobs=-1, bps_flag=False, approx_flag_global=False)
    ensemble_model.fit(X_train)
    decision_function_scores_ens = ensemble_model.decision_function(
        X_test)
    decision_function_scores_ens = aggregator_funct(
        decision_function_scores_ens, weights=vgan.proba, type="avg")

    print(vgan.subspaces.shape[0])
    return {"Dataset": dataset_name,
            "AUC": auc(y_test, decision_function_scores_ens),
            "PRAUC": average_precision_score(y_test, decision_function_scores_ens),
            "F1": f1_score(y_test, (decision_function_scores_ens > np.quantile(decision_function_scores_ens, .80)) * 1)}


def check_if_myopicity_was_uphold(dataset_name: str, gen_model_to_use="VGAN") -> tuple:
    """Given the dataset name, the function will return the p-value of the GOF test using the MMD with the recommended 
    bandwidth in [Look for the paper I don't remember the surname of the authors rn].

    Args:
        dataset_name (str): Name of the dataset as included in the json file 'datasets_file_name.json'

    Returns:
        float: p-value for the two-sampe non-parametric GoF test using the MMD with recommended bandwidth (by L2 distances)
    """
    X_train, _, _ = load_data(dataset_name)

    if gen_model_to_use == "VGAN":
        vgan = VGAN()
        vgan.load_models(Path() / "experiments" / "VGAN" /
                         f"VGAN_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    elif gen_model_to_use == "VMMD":
        vgan = VMMD()
        vgan.load_models(Path() / "experiments" / "VMMD" /
                         f"VMMD_{dataset_name}" / "models" / "generator_0.pt", ndims=X_train.shape[1])
    vgan.approx_subspace_dist()

    return vgan.check_if_myopic(X_train, bandwidth=[
        1, 0.1, 0.001, 0.0001], count=min(1000, X_train.shape[0]))["recommended bandwidth"].item(), vgan.subspaces.shape[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    dataset_name = "Ionosphere"

    auc_vgan_ens = pretrained_launch_outlier_detection_experiments(dataset_name, [
        LOF()], gen_model_to_use="VMMD")  # ,   epochs=3000, temperature=1)
    print(
        f'AUC obtained by the VGAN-based ensemble model: {print(auc_vgan_ens)}')
