from datasets.mnist import load_mnist, load_fashion_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
#from algorithms.EDSC import edesc
from sklearn import cluster
from src.cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics import normalized_mutual_info_score, clustering_accuracy
import pandas as pd
from src.modules.tools import reduce_subspaces
from pathlib import Path
from torch.utils.data import DataLoader
from src.modules.od_module import VGAN
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import time
from sklearn.ensemble import BaggingClassifier
from src.modules.tools import pretrain_autoencoder
from src.modules.network_module import Detector, Encoder, Decoder


ALGORITHMS = {
    "kmeans": cluster.KMeans(n_clusters=10), #mini batch kmeans?
    #"SSC_OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    #"Elastic": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    #"Spectral_clustering": cluster.SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=5)
}

DATASETS = {
    #"STL10": load_stl10,
    #"CIFAR100": load_cifar100,
    #"MNIST": load_mnist,
    "FASHION_MNIST": load_fashion_mnist,
    #"CIFAR10": load_cifar10
}

def visualize_reconstruction(autoencoder, data_loader, device='cuda'):
    autoencoder.eval()
    
    batch = next(iter(data_loader))[0].to(device)  # Get a batch
    with torch.no_grad():
        _, recon = autoencoder(batch)

    recon = torch.unflatten(recon, 1, batch[0].shape)

    batch = batch.cpu().permute(0, 2, 3, 1).numpy()  # Convert to (H, W, C)
    recon = recon.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(batch[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(recon[i], 0, 1))  # Clip to valid range
        axes[1, i].axis("off")

    plt.suptitle("Original vs Reconstructed Images")
    plt.savefig('decoded.png')

def plot_subspaces(images, U, dataset, shape):

    images = torch.flatten(images, 1, -1)
    # Select 20 sample images
    num_images = 20

    for i in range(num_images):
        images[i] = images[i] * U[i]

    # Define grid size
    rows, cols = 4, 5  # 4 rows, 5 columns
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
    images = torch.unflatten(images, 1, shape)
    images = images.permute(0, 2, 3, 1)
    # Plot images in the grid
    print(images.shape)
    for i, ax in enumerate(axes.flat):
        print(images[i].shape)
        ax.imshow(images[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{dataset}.png")


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

    proba = vgan.proba

    subspaces, proba = reduce_subspaces(subspaces, proba)

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
                vgan = VGAN(epochs = epoch, temperature=10, batch_size=batch_size, path_to_directory=Path()/ "experiments" / f"Example_dataset_{datetime.datetime.now()}", iternum_d=1, iternum_g=5,lr_G = lr_G, lr_D = lr_Ds)
                vgan.dataset = dataset
                print(f"CURRENTLY WORKING FOR {dataset}")

                dataset_train, dataset_test = DATASETS[dataset]()

                dataloader_train = DataLoader(dataset_train, batch_size=sample_size, shuffle=False)
                #dataloader_test = DataLoader(dataset_test, batch_size=int(sample_size / 10), shuffle=False)
                
                #autoencoder = Detector(32*32*1, 32, 1, Encoder, Decoder)
                #pretrained_ae = pretrain_autoencoder(autoencoder, dataloader_train, epochs=50, lr=0.001)

                #torch.save(pretrained_ae.encoder.state_dict(), f"./AE_Weights/encoder_weights_{dataset}.pth")
                #torch.save(pretrained_ae.decoder.state_dict(), f"./AE_Weights/decoder_weights_{dataset}.pth")

                # Call visualization function

                #visualize_reconstruction(pretrained_ae, dataloader_train)

                #exit()

                X_train, y_train = next(iter(dataloader_train))
                #X_test, y_test = next(iter(dataloader_test))

                shape = X_train.shape

                #------Preprosessing with VGAN-----#
                subspaces = vgan_training(vgan, X_train)

                plot_subspaces(X_train, subspaces, dataset, shape[1:])
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

                        X_train = torch.flatten(X_train, 1, -1)
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


                    plt.close()
                    plt.figure(figsize=(20, 6))
                    plt.plot(range(0, len(accuracys)), accuracys)
                    #plt.savefig(f'ALLACC_{name}_{dataset}_PRETRAIN_AE_WITHRELUS_ReducedSS__epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.png', dpi=300)
                    
                    plt.close()
                    plt.figure(figsize=(20, 6))
                    plt.plot(vgan.train_history["generator_loss"])
                    #plt.savefig(f'GLOSS_{dataset}_PRETRAIN_AE_WITHRELUS_ReducedSS_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.png', dpi=300)

                    plt.close()
                    plt.plot(vgan.train_history["detector_loss"])
                    plt.figure(figsize=(20, 1))
                    #plt.savefig(f'DLOSS_{dataset}_PRETRAIN_AE_WITHRELUS_ReducedSS_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.png', dpi=300)
                    plt.close()

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
        df.to_csv(f'PRETRAIN_AE_WITHCONV_CIFAR_100_ReducedSS_{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

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

        print(f"CURRENTLY WORKING FOR {dataset}")

        df.to_csv(f'ALLACC_PRETRAIN_AE_CIFAR_100_WITHCONV_ReducedSS_epoch{epoch}_s{sample_size}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

if __name__ == "__main__":

    sample_size = 2000
    batch_size = 1000
    lr_G = 0.01
    lr_D = 0.01

    run_experiment(sample_size, batch_size, lr_G, lr_D, 1500)