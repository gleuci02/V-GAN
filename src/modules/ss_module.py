from src.modules.bin import main_hics as hics
import numpy as np
import pandas as pd
import os
import umap
import sklearn.decomposition as sk
from gmd import GMD
from src.modules.tools import numeric_to_boolean
from pathlib import Path
from src.packages.Clique.Clique import get_dense_units_for_dim, get_one_dim_dense_units, get_clusters
from src.modules.network_module import MLPnet
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyod.models.base import BaseDetector
from pyod.models.lof import LOF
from sklearn import clone
# from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
# from keras.layers import Dense, Dropout, LeakyReLU
import logging
import multiprocessing
import time
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

GMD_class = GMD


def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            proc = multiprocessing.Process(
                target=func, args=(*args, return_dict), kwargs=kwargs)
            proc.start()

            start = time.time()
            while (time.time() - start) < seconds and proc.is_alive():
                pass
            if proc.is_alive():
                proc.terminate()
                raise TimeoutError("HiCS timeout!")

            proc.join()
            return return_dict.values()[0], return_dict.values()[1]
        return wrapper

    return decorator


class BaseSubspaceSelector:
    """Base class for all of the subspace selection methods. Each method is implemented in its own class, inheriting
    basic properties from the base class.

    This base class has to store a:
        .subsapce: type np.array[list[bool]]. np.ndarray of subspaces. Each subsapce is a list of boolean indiciating which feature is contained in the subsapce
        .metric: type list[int]. List of a metric/quality associated to each subspace. It will get inizialized by ones. If not used/needed, leave it like that
        .trained: type bool.  Boolean indicating wether the method has been trained or not. Useful for methods that have branching functionality
    """

    def __init__(self) -> None:
        self.subspaces = None
        self.metric = None
        self.trained = False
        pass

    def fit(self, X_train):
        self.trained = True


def decoder(x):
    x = Dense(20)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(1555)(x)
    return x


class HiCS(BaseSubspaceSelector):
    """Class for the HiCS method

    Inherits:
        BaseSubspaceSelector
    """

    def __init__(self, numRuns=100,
                 numCandidates=500, maxOutputSpaces=1000, alpha=0.1,
                 silent=False):
        super().__init__()
        # We need to load the data directly in Nim for this to work, we can not pass it as a numpy array.
        self.numRuns = numRuns
        self.numCandidates = numCandidates
        self.maxOutputSpaces = maxOutputSpaces
        self.alpha = alpha
        self.__onlySubspace = []
        self.silent = silent

    # The timeout on HiCS has to be set this way since it comes from a NIM implementation, and one can't use python's signal alarms.
    # Outsourcing the exception call to a python's signal is still the prefered choice for implementing a timeout, rather than using
    # multiprocessing timeouts on single process.
    @timeout(7200)
    def __calculate_the_subspaces(self, X_train: np.ndarray, return_dict: list):

        # Nim has troubles dealing with the numpy array direclty, so we store it first as a csv for it to read it
        np.savetxt(Path("src/modules/bin/data.csv"), X_train, delimiter=";")
        results = hics.launch(csvIn="src/modules/bin/data.csv", hasHeader=False,
                              numRuns=self.numRuns, numCandidates=self.numCandidates,
                              maxOutputSpaces=self.maxOutputSpaces, alpha=self.alpha,
                              onlySubspace=self._HiCS__onlySubspace)
        os.remove(Path("src/modules/bin/data.csv"))

        subspaces = []
        for element in results:
            subspace = []
            for item in element[1]['data']:
                if item[1] != 0:
                    subspace.append(item[1])
            subspaces.append(subspace)
        contrast = []
        for element in results:
            contrast.append(element[0])
        return_dict["contrast"] = np.array(contrast)
        return_dict["subspaces"] = np.array(numeric_to_boolean(
            subspaces, X_train.shape[1]))
        return return_dict

    def fit(self, X_train):
        """Fits the HiCS method

        Args:
            X_train (np.array): Numpy array to extract the subspaces from
        """

        self.metric, self.subspaces = self.__calculate_the_subspaces(X_train)
        self.trained = True


class CLIQUE(BaseSubspaceSelector):
    """Class for the CLIQUE method
    We select the subsapces where a cluster is fitted.

    Inherits:
        BaseSubspaceSelector
    """

    def __init__(self, xsi: float = 3, tau: float = 0.1, max_dim: int = np.Inf):
        super().__init__()
        # We need to load the data directly in Nim for this to work, we can not pass it as a numpy array.
        self.xsi = xsi
        self.tau = tau
        self.max_dim = max_dim

    def fit(self, data):
        self.subspaces = []
        dense_units = get_one_dim_dense_units(data, self.tau, self.xsi)

        # Getting 1 dimensional clusters
        clusters = get_clusters(dense_units, data, self.xsi)

        # Finding dense units and clusters for dimension > 2
        current_dim = 2
        number_of_features = np.shape(data)[1]
        while (current_dim <= number_of_features) & (len(dense_units) > 0) & (current_dim <= self.max_dim):
            dense_units = get_dense_units_for_dim(
                data, dense_units, current_dim, self.xsi, self.tau)
            i = self.subspaces.__len__() - 1
            for cluster in get_clusters(dense_units, data, self.xsi):
                clusters.append(cluster)

                if list(cluster.dimensions) in self.subspaces:
                    pass
                self.subspaces.append(list(cluster.dimensions))
            current_dim += 1
        self.subspaces = np.array(numeric_to_boolean(
            self.subspaces, number_of_features))
        self.trained = True


class GMD(BaseSubspaceSelector):
    def __init__(self, alpha=0.1, runs=100, random_state=None) -> None:
        super().__init__()
        self.alpha = alpha
        self.runs = runs
        self.random_state = random_state

    def fit(self, data):
        gmd = GMD_class(self.alpha, self.runs, self.random_state)
        gmd.fit(data)
        number_of_features = np.shape(data)[1]
        self.subspaces = list(gmd.subspaces_.values())
        self.subspaces = np.array(numeric_to_boolean(
            self.subspaces, number_of_features))
        self.metric = np.array(list(gmd.contrasts_.values()))
        self.trained = True


class ELM(BaseSubspaceSelector):

    def __init__(self,
                 batch_size=1000,
                 representation_dim=20,
                 hidden_neurons=None,
                 hidden_activation='tanh',
                 skip_connection=False,
                 n_ensemble=50,
                 max_samples=256,
                 contamination=0.1,
                 random_state=None,
                 device=None):

        super().__init__()
        self.batch_size = batch_size
        self.representation_dim = representation_dim
        self.hidden_activation = hidden_activation
        self.skip_connection = skip_connection
        self.hidden_neurons = hidden_neurons
        self.odm_trained = False

        self.n_ensemble = n_ensemble
        self.max_samples = max_samples

        self.random_state = random_state
        self.device = device

        self.minmax_scaler = None

        # create default calculation device (support GPU if available)
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        # set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            np.random.seed(self.random_state)

        # default values for the amount of hidden neurons
        if self.hidden_neurons is None:
            self.hidden_neurons = [500, 100]

    def _deep_representation(self, net, X):
        x_reduced = []

        with torch.no_grad():
            loader = DataLoader(X, batch_size=self.batch_size,
                                drop_last=False, pin_memory=True,
                                shuffle=False)
            for batch_x in loader:
                batch_x = batch_x.float().to(self.device)
                batch_x_reduced = net(batch_x)
                x_reduced.append(batch_x_reduced)

        x_reduced = torch.cat(x_reduced).data.cpu().numpy()
        x_reduced = StandardScaler().fit_transform(x_reduced)
        x_reduced = np.tanh(x_reduced)
        return x_reduced

    def fit(self, X_train):
        n_samples, n_features = X_train.shape[0], X_train.shape[1]

        # conduct min-max normalization before feeding into neural networks
        self.minmax_scaler = MinMaxScaler()
        self.minmax_scaler.fit(X_train)
        X_train = self.minmax_scaler.transform(X_train)

        # prepare neural network parameters
        network_params = {
            'n_features': n_features,
            'n_hidden': self.hidden_neurons,
            'n_output': self.representation_dim,
            'activation': self.hidden_activation,
            'skip_connection': self.skip_connection
        }

        ensemble_seeds = np.random.randint(0, 100000, self.n_ensemble)
        self.net_lst = []
        for i in range(self.n_ensemble):
            # instantiate network class and seed random seed
            net = MLPnet(**network_params).to(self.device)
            torch.manual_seed(ensemble_seeds[i])

            # initialize network parameters
            for name, param in net.named_parameters():
                if name.endswith('weight'):
                    torch.nn.init.normal_(param, mean=0., std=1.)

            self.net_lst.append(net)
        self.trained = True

    def fit_odm(self, X_train: np.ndarray, base_odm: BaseDetector = LOF()):
        assert self.trained, "The ELMs have not been initialized. Run self.fit"
        odm_list = []
        for i, net in enumerate(self.net_lst):
            logger.info(f"Training ELM {i+1} out of {self.net_lst.__len__()}")
            x_reduced = self._deep_representation(net, X_train)

            odm = clone(base_odm)
            odm.fit(x_reduced)
            odm_list.append(odm)
        self.odm_trained = True
        self.odm_list = odm_list

    def decision_function_odm(self, X_test):
        assert self.trained, "The ELMs have not been initialized. Run self.fit"
        assert self.odm_trained, "No ODM has been trained"

        decision_function = []
        for i, net in enumerate(self.net_lst):
            odm = self.odm_list[i]
            x_reduced = self._deep_representation(net, X_test)

            decision_function.append(odm.decision_function(x_reduced).tolist())

        return np.transpose(np.array(decision_function))


class CAE(BaseSubspaceSelector):
    def __init__(self, K: int = 20, output_function=None,
                 num_epochs: int = 300, batch_size: int = None,
                 learning_rate: float = .001, start_temp: float = 10.0,
                 min_temp: float = .1, tryout_limit: int = 5) -> None:
        super().__init__()
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit

    def give_default_output_function(self, X_train_dims):
        def output_function(x):
            x = Dense(int(X_train_dims/2.5))(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.1)(x)
            x = Dense(int(X_train_dims/2.5))(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.1)(x)
            x = Dense(int(X_train_dims))(x)
            return x
        return output_function

    def fit(self, X_train: np.ndarray):
        if self.output_function == None:
            self.__output_function__ = self.give_default_output_function(
                X_train.shape[1])
        else:
            self.__output_function__ = self.output_function

        if self.K > X_train.shape[1]/2:
            self.K = int(X_train.shape[1]/2)

        selector = ConcreteAutoencoderFeatureSelector(self.K, self.__output_function__, self.num_epochs,
                                                      self.batch_size, self.learning_rate,
                                                      self.start_temp, self.min_temp, self.tryout_limit)
        selector.fit(X_train, Y=None)
        self.subspaces = np.array(
            [(selector.get_support(indices=False) > 0).tolist()])

        self.trained = True


class PCA(BaseSubspaceSelector):
    def __init__(self, random_state=None, select_up_to_ratio=.95) -> None:
        super().__init__()
        self.random_state = random_state
        self.select_up_to_ratio = .9

    def fit(self, X_train):
        pca = sk.PCA()
        pca.fit(X_train)

        self.metric = np.array(pca.explained_variance_ratio_)
        last_selected_component = 0
        selected_components_var_sum = self.metric[last_selected_component]

        while selected_components_var_sum < self.select_up_to_ratio:
            last_selected_component += 1
            selected_components_var_sum += self.metric[last_selected_component]
        self.__transform = pca.transform
        self.__last_selected_component = last_selected_component
        self.trained = True

    def transform(self, X_sample):
        X_transformed = self.__transform(X_sample)
        return X_transformed[:, :self.__last_selected_component + 1]

    def fit_odm(self, X_train: np.ndarray, base_odm: BaseDetector = LOF()):
        self.__odm = base_odm
        self.__odm.fit(self.transform(X_train))

    def decision_function_odm(self, X_test):
        return self.__odm.decision_function(self.transform(X_test))


class UMAP(BaseSubspaceSelector):
    def __init__(self, n_components=None, n_neighbors=15, random_state=None) -> None:
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit(self, X_train):
        if self.n_components == None:
            component_array = [int(i) for i in np.linspace(
                5, max(int(X_train.shape[1]), 7), 5)]
            component_array.sort()
            dim_red = umap.UMAP(n_components=component_array[0],
                                n_neighbors=self.n_neighbors, random_state=self.random_state, verbose=True)
            dim_red.fit(X_train)
            X_recon = dim_red.inverse_transform(
                dim_red.transform(X_train))
            current_rec_loss = np.linalg.norm(X_train-X_recon, axis=1).sum()
            print(f"Reconstruction error: {current_rec_loss}")
            self.__transform = dim_red.transform
            for self.n_components in component_array[1:]:
                dim_red = umap.UMAP(n_components=self.n_components,
                                    n_neighbors=self.n_neighbors, random_state=self.random_state, verbose=True)
                dim_red.fit(X_train)
                X_recon = dim_red.inverse_transform(
                    dim_red.transform(X_train))
                rec_loss = np.linalg.norm(X_train-X_recon, axis=1).sum()
                print(f"Reconstruction error: {rec_loss}")
                if rec_loss < current_rec_loss:
                    print(f"Reconstruction error lower than previous")
                    self.__transform = dim_red.transform
                    current_rec_loss = rec_loss
                else:
                    print(f"Best reconstruction error found")
                    break
        else:
            dim_red = umap.UMAP(n_components=min([int(X_train.shape[1]), 100]),
                                n_neighbors=self.n_neighbors, random_state=self.random_state, verbose=True, n_jobs=20)

            dim_red.fit(X_train)
            self.__transform = dim_red.transform

    def transform(self, X_sample):
        return self.__transform(X_sample)

    def fit_odm(self, X_train: np.ndarray, base_odm: BaseDetector = LOF()):
        self.__odm = base_odm
        self.__odm.fit(self.transform(X_train))

    def decision_function_odm(self, X_test):
        return self.__odm.decision_function(self.transform(X_test))
