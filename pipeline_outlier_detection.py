from src.modules.od_module import VGAN
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD
from pyod.models.feature_bagging import FeatureBagging
from pathlib import Path
import datetime
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.metrics import roc_auc_score as auc
from sel_suod.models.base import sel_SUOD
import itertools
from src.modules.tools import numeric_to_boolean, aggregator_funct
from sklearn.preprocessing import label_binarize
from joblib.externals.loky import get_reusable_executor
from data.get_datasets import load_data
import json
from pyod.utils.utility import generate_bagging_indices
import random
from sklearn.metrics import average_precision_score, f1_score
import os
from outlier_detection import launch_outlier_detection_experiments, pretrained_launch_outlier_detection_experiments, check_if_myopicity_was_uphold
import logging
from colorama import Fore
from sklearn.base import clone
import src.modules.ss_module as ss
import time
import signal

logger = logging.getLogger(__name__)


def timeout_call(signum, frame):
    raise TimeoutError("Time excedeed the intended execution time")


def pipeline_outlier_detection_ens_od(datasets: list, outlier_detection_models: list = None, experimental_settings: np.array = None,
                                      base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000, 20, 30, 33, 90, 10], base_subspace_selector: ss.BaseSubspaceSelector = ss.HiCS()):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    time_df = pd.DataFrame(
        {"Dataset": [], "Time": []}
    )
    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} with {Fore.CYAN}{base_subspace_selector.__class__.__name__}{Fore.RESET}, number {j+1} out of {len(base_methods)}")

            assert base_subspace_selector.__class__.__name__ not in [
                "ELM", "PCA", "UMAP"], "Subspace selector not accepted"
            X_train, X_test, y_test = load_data(dataset)
            subspace_selector = base_subspace_selector
            path_subspace = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
                "subspaces" / \
                f"{subspace_selector.__class__.__name__}_{dataset}.npz"
            path_metric = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / "metrics" / \
                f"{subspace_selector.__class__.__name__}_{dataset}.npz"
            path_results = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
                f"Results_{subspace_selector.__class__.__name__}_{
                    [method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv"
            path_times = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
                f"Times_{subspace_selector.__class__.__name__}_all_datasets.csv"

            if os.path.isfile(path_subspace) and subspace_selector.__class__.__name__ != "CAE":
                try:
                    subspace_selector.subspaces = np.load(path_subspace)[
                        "arr_0"]
                except:
                    subspace_selector.subspaces = None
                try:
                    subspace_selector.metric = np.load(path_metric)["arr_0"]
                except:
                    subspace_selector.metric = None
            else:
                try:
                    signal.signal(signal.SIGALRM, handler=timeout_call)
                    tic = time.time()
                    signal.alarm(7200)
                    subspace_selector.fit(X_train)
                    toc = time.time()
                    signal.alarm(0)

                except TimeoutError:
                    signal.alarm(0)
                    toc = time.time()
                    logger.warning(
                        f"Timeout for ss method {subspace_selector.__class__.__name__} in {dataset}!")
                    results_dict = {"Dataset": dataset,
                                    "AUC": "TIME OUT",
                                    "PRAUC": "TIME OUT",
                                    "F1": "TIME OUT"}
                    time_dict = {"Dataset": dataset, "Time": 7200}
                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                    time_df = time_df._append(time_dict, ignore_index=True)
                    results_df.to_csv(path_results)
                    time_df.to_csv(path_times)
                    break
                time_dict = {"Dataset": dataset, "Time": toc-tic}
                time_df = time_df._append(time_dict, ignore_index=True)
                time_df.to_csv(path_times)
                np.savez(path_subspace,
                         subspace_selector.subspaces)
                np.savez(path_metric,
                         subspace_selector.metric)

            for seed in seeds:
                try:
                    random.seed(seed)
                    ensemble_method = sel_SUOD(base_estimators=[clone(
                        base_method)], subspaces=subspace_selector.subspaces, n_jobs=48, bps_flag=False, approx_flag_global=False)
                    ensemble_method.fit(X_train)
                    decision_function_scores = ensemble_method.decision_function(
                        X_test)
                    decision_function_scores = aggregator_funct(
                        decision_function_scores, type="avg", weights=subspace_selector.metric)
                    results_dict = {"Dataset": dataset,
                                    "AUC": auc(y_test, decision_function_scores),
                                    "PRAUC": average_precision_score(y_test, decision_function_scores),
                                    "F1": f1_score(y_test, (decision_function_scores > np.quantile(decision_function_scores, .80)) * 1)}

                    results_dict["Method"] = base_method.__class__.__name__

                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                except:
                    logger.warning(
                        f"Error found during exection in dataset: {dataset} with method {base_method.__class__.__name__}")
                    continue

                results_df.to_csv(path_results)


def pipeline_outlier_detection_ELM_od(datasets: list, outlier_detection_models: list = None, experimental_settings: np.array = None,
                                      base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000, 20, 30, 33, 90, 10]):
    """Pipeline for the outlier detection experiments

    This function will run the outlier detection experiments in a collection of datasets and for a group of outlier detection models.
    Additionally, the funcition is capable of, given the existence of a version of VGAN/VMMD, load it and use it for its own porpuse

    Args:
        datasets (list): list of datasets to run the experiments in
        outlier_detection_models (list): list of outlier detection models to run the experiments withs (besides vgan ensemble)
        base_methods (list): List of base_estimators to employ as the basis of the ensemble used in VGAN. It does NOT function as in SUOD and sel_SUOD,
        but rather as FeatureBagging, where every model introduced here will be used in SEPARATE homogeneous ensembles.
    """
    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    time_df = pd.DataFrame(
        {"Dataset": [], "Time": []}
    )
    path_results = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
        f"Results_ELM_{
            [method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv"
    path_times = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
        f"Times_ELM_all_datasets.csv"

    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} with {Fore.CYAN}ELM{Fore.RESET}, number {j+1} out of {len(base_methods)}")
            for seed in seeds:

                X_train, X_test, y_test = load_data(dataset)
                subspace_selector = ss.ELM(random_state=seed)
                tic = time.time()
                subspace_selector.fit(X_train)
                toc = time.time()

                time_dict = {"Dataset": dataset, "Time": toc-tic}
                time_df = time_df._append(time_dict, ignore_index=True)
                time_df.to_csv(path_times)

                try:
                    random.seed(seed)
                    subspace_selector.fit_odm(
                        X_train=X_train, base_odm=base_method)
                    decision_function_scores = subspace_selector.decision_function_odm(
                        X_test)
                    decision_function_scores = aggregator_funct(
                        decision_function_scores, type="avg", weights=subspace_selector.metric)
                    results_dict = {"Dataset": dataset,
                                    "AUC": auc(y_test, decision_function_scores),
                                    "PRAUC": average_precision_score(y_test, decision_function_scores),
                                    "F1": f1_score(y_test, (decision_function_scores > np.quantile(decision_function_scores, .80)) * 1)}

                    results_dict["Method"] = base_method.__class__.__name__

                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                except:
                    logger.warning(
                        f"Error found during exection in dataset: {dataset} with method {base_method.__class__.__name__}")
                    continue

                results_df.to_csv(path_results)


def pipeline_outlier_detection_dimred_od(datasets: list, base_subspace_selector: ss.BaseSubspaceSelector,  outlier_detection_models: list = None, experimental_settings: np.array = None,
                                         base_methods: list = [], seeds=[777, 1234, 12345, 000, 1000, 20, 30, 33, 90, 10]):

    results_df = pd.DataFrame(
        {"Dataset": [], "Method": [], "AUC": [], "PRAUC": [], "F1": []})
    time_df = pd.DataFrame(
        {"Dataset": [], "Time": []}
    )
    path_results = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
        f"Results_{base_subspace_selector.__class__.__name__}_{
            [method.__class__.__name__ for method in base_methods].__repr__()}_all_datasets.csv"
    path_times = Path() / "experiments" / "Outlier_Detection" / "COMPETITORS" / \
        f"Times_{base_subspace_selector.__class__.__name__}_all_datasets.csv"

    for i, dataset in enumerate(datasets):
        for j, base_method in enumerate(base_methods):
            logger.info(
                f"Running dataset {Fore.CYAN}{dataset}{Fore.RESET}, number {i+1} out of {datasets.__len__()} using method: {base_method.__class__.__name__} with {Fore.CYAN}{base_subspace_selector.__class__.__name__}{Fore.RESET}, number {j+1} out of {len(base_methods)}")
            for seed in seeds:

                X_train, X_test, y_test = load_data(dataset)
                subspace_selector = base_subspace_selector
                subspace_selector.random_state = seed
                tic = time.time()
                subspace_selector.fit(X_train)
                toc = time.time()

                time_dict = {"Dataset": dataset, "Time": toc-tic}
                time_df = time_df._append(time_dict, ignore_index=True)
                time_df.to_csv(path_times)

                try:
                    random.seed(seed)
                    subspace_selector.fit_odm(
                        X_train=X_train, base_odm=base_method)
                    decision_function_scores = subspace_selector.decision_function_odm(
                        X_test)
                    results_dict = {"Dataset": dataset,
                                    "AUC": auc(y_test, decision_function_scores),
                                    "PRAUC": average_precision_score(y_test, decision_function_scores),
                                    "F1": f1_score(y_test, (decision_function_scores > np.quantile(decision_function_scores, .80)) * 1)}

                    results_dict["Method"] = base_method.__class__.__name__

                    results_df = results_df._append(
                        results_dict, ignore_index=True)
                except:
                    logger.warning(
                        f"Error found during exection in dataset: {dataset} with method {base_method.__class__.__name__}")
                    continue

                results_df.to_csv(path_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    datasets = ["20news_0",
                "MVTec-AD_bottle",
                "InternetAds",
                "agnews_0",
                "amazon",
                "imdb",
                "yelp",
                "MNIST-C_brightness",
                "CIFAR10_0",
                "FashionMNIST_0",
                "SVHN_0",
                "speech",
                "musk",
                "mnist",
                "optdigits",
                "SpamBase",
                "landsat",
                "satellite",
                "satimage-2",
                "Ionosphere",
                "WPBC",
                "letter",
                "WDBC",
                "fault",
                "annthyroid",
                "cardio",
                "Cardiotocography",
                "Waveform",
                "Hepatitis",
                "Lymphography",
                "pendigits",
                "wine",
                "vowels",
                "PageBlocks",
                "breastw",
                "Stamps",
                "WBC",
                "Pima",
                "yeast",
                "thyroid",
                "vertebral",
                "Wilt",
                "vertebral",
                "Wilt"]

for base_method in [LOF(), KNN(), CBLOF(), ECOD(), COPOD()]:
    # pipeline_outlier_detection_ens_od(
    #    datasets, base_methods=[base_method], base_subspace_selector=ss.HiCS())
    # pipeline_outlier_detection_ens_od(
    #    datasets, base_methods=[base_method], base_subspace_selector=ss.GMD())
    # pipeline_outlier_detection_ens_od(
    #    datasets, base_methods=[
    #        base_method], base_subspace_selector=ss.CAE()
    # )
    # pipeline_outlier_detection_ens_od(
    #    datasets, base_methods=[base_method], base_subspace_selector=ss.CLIQUE())
    # pipeline_outlier_detection_ELM_od(datasets, base_methods=[base_method])
    pipeline_outlier_detection_dimred_od(
        datasets, base_subspace_selector=ss.PCA(), base_methods=[base_method])
    # pipeline_outlier_detection_dimred_od(
    #    datasets, base_subspace_selector=ss.UMAP(n_components=-1), base_methods=[base_method])
