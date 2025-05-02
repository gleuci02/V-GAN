from datasets.mnist import load_mnist, load_fashion_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
from datasets.caltech import load_caltech101
#from algorithms.EDSC import edesc
from sklearn import cluster
#from src.cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics import normalized_mutual_info_score, clustering_accuracy
import pandas as pd
from src.modules.tools import reduce_subspaces
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from src.modules.od_module import VGAN, VMMD
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import time
from sklearn.ensemble import BaggingClassifier

path = "results/results_Caltech/"

ALGORITHMS = {
    "kmeans": cluster.KMeans(n_clusters=10), #mini batch kmeans?
    #"SSC_OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    #"Elastic": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    #"Spectral_clustering": cluster.SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=5)
}

DATASETS = {
    #"CALTECH101": load_caltech101,
    #"STL10": load_stl10,
    #"CIFAR100": load_cifar100,
    "MNIST": load_mnist,
    #"FASHION_MNIST": load_fashion_mnist,
    #"CIFAR10": load_cifar10
}

def plot_subspaces(images, U, dataset, shape): #TODO Use UMAP ???
    # Select 20 sample images
    num_images = 20
    rows, cols = 4, 5  # 4 rows, 5 columns
    
    U = torch.unflatten(U, 1, shape[1:])

    selected_images = []
    for i in range(num_images):
        col_idx = i % cols
        if col_idx < len(U):
            images[i] = images[i] * U[col_idx]
            #selected_images.append(images[i][:, U[col_idx]])
    
    #selected_images = torch.stack(selected_images)

    # Reshape images for plotting
    #selected_images = selected_images.permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype("float32")
    
    # Create subplots
    fig, axes = plt.subplots(rows + 1, cols, figsize=(10, 10))
    
    # Plot U on the top row
    for j in range(cols):
        if j < len(U):
            axes[0, j].imshow(U[j], cmap="gray")
        axes[0, j].axis("off")
    
    # Plot images in the grid
    for i in range(num_images):
        row, col = divmod(i, cols)
        ax = axes[row + 1, col]
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(f"{path}{dataset}.png")
    print(f"{path}{dataset}.png")

def generate_clustering_ensemble(clusterings, amount_cluster):
    n_samples = clusterings.shape[1]
    co_matrix = np.zeros((n_samples, n_samples))
    
    for clustering in clusterings:
        for i in range(n_samples):
            for j in range(i, n_samples):
                if clustering[i] == clustering[j]:
                    co_matrix[i, j] += 1
                    co_matrix[j, i] += 1

    co_matrix /= co_matrix.diagonal()  # Normalize by number of clusterings

    spectral = cluster.SpectralClustering(amount_cluster, affinity="precomputed")
    final_clusters = spectral.fit_predict(co_matrix)

    return final_clusters

def vgan_training(vgan, X_train):

    vgan.fit(X_train)

    vgan.approx_subspace_dist(500, add_leftover_features=True)

    subspaces = vgan.subspaces

    print(subspaces.shape)

    subspaces = subspaces.reshape(subspaces.shape[0], -1)

    proba = vgan.proba

    subspaces, proba = reduce_subspaces(subspaces, proba)

    print(len(subspaces))

    tup = zip(subspaces, proba)

    subspaces = sorted(tup, key=lambda tup: tup[1], reverse=True)[:30]

    subspaces = [x[0] for x in subspaces]

    return torch.tensor(subspaces).cpu()

def feature_bagging(algo, X_train, X_test, y_train):
    bagging = BaggingClassifier(estimator=algo, n_estimators=30)

    bagging.fit(X_train, y_train)

    y_pred = bagging.predict(X_test)

    return y_pred

def feature_bagging_experiment(batch_size):

    datasets = []
    total_times = []
    nmis = []
    accuracys = []    # 5 datasets
    names = []

    for i, dataset in enumerate(DATASETS):
        dataset_train, dataset_test = DATASETS[dataset]()

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        X_train, y_train = next(iter(dataloader_train))
        X_test, y_test = next(iter(dataloader_test))

        for name in ALGORITHMS:

            algorithm = ALGORITHMS[name]

            if dataset == "CIFAR100":
                    algorithm.set_params(n_clusters=100)
            
            starting_time = time.time()
            y_pred = feature_bagging(algo=algorithm, X_train=X_train, X_test=X_test, y_train=y_train)
            end_time = time.time()
            total_time = end_time - starting_time

            nmi = normalized_mutual_info_score(y_test, y_pred)
            acc = clustering_accuracy(y_test, y_pred)

            nmis.append(nmi)
            accuracys.append(acc)
            datasets.append(dataset)
            names.append(name)
            total_times.append(total_time)

    df = pd.DataFrame({
            "DATABASE": datasets,
            "ALGORITHM": names,
            "TIME TAKEN": total_times,
            "NMI": nmis,
            "ACC": accuracys
        })

    df.to_csv(f'Feature_bagging.csv')


def run_experiment(sample_size, batch_size, lr_G, lr_Ds, epoch):
       
        datasets = []
        total_times = []
        nmis = []
        accuracys = []    # 5 datasets
        names = []

        total_times_ensemble = []
        names_ensemble = []
        datasets_ensemble = []
        acc_ensemble = []
        nmi_ensemble = []
        results = []
        bandwiths = []

        # Load dataset
        for i, dataset in enumerate(DATASETS):
                vgan = VMMD(epochs=epoch, path_to_directory=Path() / "experiments" /f"Example_normal_{datetime.datetime.now()}_vmmd", lr=lr_G)
                    
                print(f"CURRENTLY WORKING FOR {dataset}")

                dataset_train, dataset_test = DATASETS[dataset]()

                train_indices = [i for i, (_, label) in enumerate(dataset_train) if label == 1] #17
                test_indices = [i for i, (_, label) in enumerate(dataset_test) if label == 1]

                # Create filtered datasets
                filtered_train = Subset(dataset_train, train_indices)
                filtered_test = Subset(dataset_test, test_indices)

                dataloader_train = DataLoader(filtered_train, batch_size=sample_size, shuffle=False)
                #dataloader_test = DataLoader(dataset_test, batch_size=int(sample_size / 10), shuffle=False)

                X_train, y_train = next(iter(dataloader_train))
                shape = X_train.shape
                #X_train = torch.flatten(X_train, 1, -1)
                #X_test, y_test = next(iter(dataloader_test))

                #------Preprosessing with VGAN-----#
                subspaces = vgan_training(vgan, X_train)

                plot_subspaces(X_train, subspaces, dataset, (shape[1], shape[2], shape[3]))

                plt.clf()
                plt.figure(figsize=(20, 6))
                plt.plot(vgan.train_history["generator_loss"])
                plt.show()
                plt.savefig(f'{path}VGAN_GLOSS_{dataset}_ReducedSS_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D.png', dpi=300)
                plt.clf()

                exit()
                #------End of Preprosessing with VGAN-----#

                for name in ALGORITHMS:

                    clusterings = []

                    algorithm = ALGORITHMS[name]

                    if dataset == "CIFAR100":
                         algorithm.set_params(n_clusters=100)

                    amount_cluster = algorithm.get_params()["n_clusters"]

                    for subspace in subspaces:

                        starting_time = time.time()
                        # Initialize algorithm

                        X_new = X_train * subspace

                        y_pred = algorithm.fit_predict(X_new)

                        clusterings.append(y_pred)

                        nmi = normalized_mutual_info_score(y_train, y_pred)
                        acc = clustering_accuracy(y_train, y_pred)
                        
                        end_time = time.time()
                        total_time = end_time - starting_time
                        
                        datasets.append(dataset)
                        names.append(name)
                        total_times.append(total_time)
                        nmis.append(nmi)
                        accuracys.append(acc)

                    clusterings = np.array(clusterings)
                    final_cluster = generate_clustering_ensemble(clusterings, amount_cluster)

                    nmi = normalized_mutual_info_score(y_train, final_cluster)
                    acc = clustering_accuracy(y_train, final_cluster)

                    X_new = np.array(X_new)
                    X_new = torch.as_tensor(X_new, device=torch.device('cpu'))

                    result, bandwith = vgan.check_if_myopic(X_train, [vgan.bandwidth.cpu()], len(X_train))
                    #df.to_csv(f'FB_{name}{dataset}check_if_myopic.csv')

                    bandwiths.append(bandwith)
                    results.append(result)
                    datasets_ensemble.append(dataset)
                    names_ensemble.append(name)
                    total_times_ensemble.append(total_time)
                    acc_ensemble.append(acc)
                    nmi_ensemble.append(nmi)

                    

        df = pd.DataFrame({
            "DATABASE": datasets_ensemble,
            "ALGORITHM": names_ensemble,
            "TIME TAKEN": total_times_ensemble,
            "NMI": nmi_ensemble,
            "ACC": acc_ensemble,
            "PValue": results,
            "bandwidth": bandwiths
        })

        #df.to_csv(f'FB_Ensemble_ReducedSS_Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')
        df.to_csv(f'{path}VGAN_{epoch}_s{sample_size}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

        #final_cluster = generate_clustering_ensemble(clusterings, amount_cluster)

        acc = clustering_accuracy(y_train, final_cluster)
        nmi = normalized_mutual_info_score(y_train, final_cluster)

        df = pd.DataFrame({
            "DATABASE": datasets,
            "ALGORITHM": names,
            "TIME TAKEN": total_times,
            "NMI": nmis,
            "ACC": accuracys
        })

        df.to_csv(f'{path}VGAN_ALLACC_epoch{epoch}_s{sample_size}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

if __name__ == "__main__":

    sample_size = 4000
    batch_size = 2000
    lr_G = 0.007
    lr_D = 0.01

    run_experiment(sample_size, batch_size, lr_G, lr_D, 1200)