from datasets.mnist import load_mnist, load_fashion_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
from datasets.mvtechAD import load_fruit_jelly, load_vials
from datasets.caltech import load_caltech101
#from algorithms.EDSC import edesc
from sklearn import cluster
from src.cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics import normalized_mutual_info_score, clustering_accuracy
import pandas as pd
from src.modules.tools import reduce_subspaces
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from src.modules.od_module import VGAN, VMMD
import matplotlib.pyplot as plt
from src.cluster.SNNDPC import SNNDPC
import numpy as np
import torch
import datetime
import time

path = "results/test/" #results_avg2

ALGORITHMS = {
    "kmeans": cluster.KMeans(n_clusters=10), #mini batch kmeans?
    #"SSC_OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    #"Elastic": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    #"Spectral_clustering": cluster.SpectralClustering(n_clusters=10, affinity='nearest_neighbors'),
    #"SNNDPC": SNNDPC(n_clusters=10, k=10), #Shared-Nearest-Neighbor-Based Clustering by Fast Search and Find of Density Peaks
}

DATASETS = {
    #"MNIST": load_mnist,
    #"FASHION_MNIST": load_fashion_mnist,
    #"CIFAR10": load_cifar10,
    "STL10": load_stl10,
    "CIFAR100": load_cifar100,
    "CALTECH101": load_caltech101,
    #"vials": load_vials,
    #"fruit_jelly": load_fruit_jelly,
}

def full_space(sample_size, batch_size, lr_G, lr_Ds, epoch, seed):
        timenow = datetime.datetime.now()
        datasets = []
        total_times = []
        nmis = []
        names = []
        accuracys = []

        # Load dataset
        for i, dataset in enumerate(DATASETS):
                #torch.manual_seed(seed)
                #np.random.seed(seed)
                print(f"CURRENTLY WORKING FOR {dataset}")

                dataset_train, dataset_test = DATASETS[dataset]()

                dataloader_train = DataLoader(dataset_train, batch_size=sample_size, shuffle=True)

                X_train, y_train = next(iter(dataloader_train))

                print(X_train.shape)

                X_train = X_train.flatten(1, -1)
                
                print(sample_size)
                print(batch_size)

                for name in ALGORITHMS:
                    
                    algorithm = ALGORITHMS[name]

                    if dataset == "CIFAR100":
                        algorithm.set_params(n_clusters=100)
                    elif dataset == "CALTECH101":
                        algorithm.set_params(n_clusters=101)

                    starting_time = time.time()
                    # Initialize algorithm

                    print(dataset)
                    print(name)
                    y_pred = algorithm.fit_predict(X_train)

                    nmi = normalized_mutual_info_score(y_train, y_pred)
                    acc = clustering_accuracy(y_train, y_pred)
                    
                    end_time = time.time()
                    total_time = end_time - starting_time
                        
                    datasets.append(dataset)
                    names.append(name)
                    total_times.append(total_time)
                    nmis.append(nmi)
                    accuracys.append(acc)

        df = pd.DataFrame({
            "DATABASE": datasets,
            "ALGORITHM": names,
            "TIME TAKEN": total_times,
            "NMI": nmis,
            "ACC": accuracys
        })

        print(f"CURRENTLY WORKING FOR {dataset}")

        print(f'{path}{timenow}_full_space_s{sample_size}.csv')
        df.to_csv(f'{path}{timenow}_full_space_s{sample_size}.csv')


def visualize_reconstruction(autoencoder, data_loader, device='cuda'):
    autoencoder.eval()
    
    batch = next(iter(data_loader))[0].to(device)  # Get a batch
    with torch.no_grad():
        _, recon = autoencoder(batch)

    print(recon.shape)
    recon = torch.unflatten(recon, 1, batch[0].shape)

    batch = batch.cpu().permute(0, 2, 3, 1).numpy()  # Convert to (H, W, C)
    recon = recon.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(batch[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray")  # Clip to valid range
        axes[1, i].axis("off")

    plt.suptitle("Original vs Reconstructed Images")
    plt.savefig('decoded.pdf')

def plot_subspaces(vgan, images, U, dataset, shape):
    # Select 20 sample images
    num_images = 20
    rows, cols = 4, 5  # 4 rows, 5 columns 

    print(U.shape)

    newShape = int(np.sqrt(U.shape[1]))

    U = U.unflatten(1, (newShape, newShape))

    U = torch.Tensor(U)

    for i in range(num_images):
        col_idx = i % cols
        if col_idx < len(U):
            proj_enc = images.to('cuda') * U[col_idx].to('cuda')
            images[i] = proj_enc[i]

    # Reshape images for plotting
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy().astype("float32")
    
    # Create subplots
    fig, axes = plt.subplots(rows + 1, cols, figsize=(10, 10))
    
    # Plot U on the top row
    for j in range(cols):
        if j < len(U):
            axes[0, j].imshow(U[j], cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        axes[0, j].axis("off")
    
    # Plot images in the grid
    for i in range(num_images):
        row, col = divmod(i, cols)
        ax = axes[row + 1, col]
        ax.imshow(images[i], cmap="gray", interpolation="nearest")
        #ax.imshow(U[i], cmap="gray", interpolation="nearest")
        ax.axis("off")
    
    plt.tight_layout()
    timenow = datetime.datetime.now()
    plt.savefig(f"{path}{timenow}{dataset}.pdf")
    print(f"{path}{dataset}.pdf")


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
    #latent_size = max(int(X_train.flatten(1, -1).shape[1]/16), 1)
    #vgan.load_models("./experiments/Example_normal_2025-03-16 14:17:39.055543_vmmd/models/generator_0.pt", X_train.shape[2], latent_size)
    vgan.approx_subspace_dist(500, add_leftover_features=False)

    subspaces = vgan.subspaces

    proba = vgan.proba

    print(subspaces.shape)
    subspaces, proba = reduce_subspaces(subspaces, proba)

    tup = zip(subspaces, proba)

    subspaces = sorted(tup, key=lambda tup: tup[1], reverse=True)[:30]

    subspaces = [x[0] for x in subspaces]

    return torch.tensor(subspaces).cpu()

def run_experiment(sample_size, batch_size, lr_G, lr_Ds, epoch, seed):
        timenow = datetime.datetime.now()
        datasets = []
        total_times = []
        nmis = []
        names = []
        accuracys = []

        total_times_ensemble = []
        names_ensemble = []
        datasets_ensemble = []
        acc_ensemble = []
        nmi_ensemble = []

        # Load dataset
        for i, dataset in enumerate(DATASETS):
                vgan = VMMD(epochs=epoch, path_to_directory=Path() / "experiments" /f"Example_normal_{datetime.datetime.now()}_vmmd", lr=lr_G)
                vgan.seed = seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                vgan.dataset = dataset
                print(f"CURRENTLY WORKING FOR {dataset}")

                dataset_train, dataset_test = DATASETS[dataset]()

                # Get indices where the label is 1 (Pants)
                train_indices = [i for i, (_, label) in enumerate(dataset_train) if label == 17] #17
                test_indices = [i for i, (_, label) in enumerate(dataset_test) if label == 17]

                # Create filtered datasets
                filtered_train = Subset(dataset_train, train_indices)
                filtered_test = Subset(dataset_test, test_indices)

                dataloader_train = DataLoader(dataset_train, batch_size=sample_size, shuffle=False)

                #dataloader_test = DataLoader(dataset_test, batch_size=int(sample_size / 10), shuffle=False)

                X_train, y_train = next(iter(dataloader_train))
                
                shape = X_train.shape

                #------Preprosessing with VGAN-----#
                subspaces = vgan_training(vgan, X_train)
                print(shape)

                #Feature Bagging
                ##n_features = X_train.shape[2] * X_train.shape[3]
                ##feature_masks = [np.random.choice([0, 1], size=n_features, p=[0.5, 0.5]) for _ in range(30)]
                ##feature_masks = torch.Tensor(feature_masks)

                test = vgan.check_if_myopic(X_train.detach().numpy(), [vgan.bandwidth], len(X_train))
                test.to_csv(f'{path}ifmyopic{dataset}.csv')
                print(f'{path}ifmyopic{dataset}.csv')
                print(test)

                ##continue

                plt.clf()
                plt.figure(figsize=(20, 6))
                plt.plot(vgan.train_history["generator_loss"])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Generator Loss over Epochs')
                plt.legend()
                print(f'{path}{timenow}VGAN_GLOSS_{dataset}_ReducedSS_Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D.pdf')
                plt.savefig(f'{path}{timenow}VGAN_GLOSS_{dataset}_ReducedSS_Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D.pdf', dpi=300)
                
                plot_subspaces(vgan, X_train, subspaces, dataset, shape[1:])
                
                print(vgan.generator)
                print(lr_G)
                print(epoch)
                print(sample_size)
                print(batch_size)

                ##return
                #continue
                #------End of Preprosessing with VGAN-----#

                for name in ALGORITHMS:
                    
                    clusterings = []

                    algorithm = ALGORITHMS[name]

                    if dataset == "CIFAR100":
                        algorithm.set_params(n_clusters=100)
                    elif dataset == "CALTECH101":
                        algorithm.set_params(n_clusters=101)

                    amount_cluster = algorithm.get_params()["n_clusters"]

                    for subspace in subspaces:
                    #for subspace in feature_masks:
                        
                        starting_time = time.time()
                        # Initialize algorithm

                        subspace = torch.unflatten(subspace, 0, shape[2:])
                        X_new = X_train * subspace
                        X_new = torch.flatten(X_new, 1, -1)

                        print(dataset)
                        print(name)
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

                    #result, bandwith = vgan.check_if_myopic(X_train, [vgan.bandwidth.cpu()], len(X_train))
                    #df.to_csv(f'FB_{name}{dataset}check_if_myopic.csv')

                    datasets_ensemble.append(dataset)
                    names_ensemble.append(name)
                    total_times_ensemble.append(total_time)
                    acc_ensemble.append(acc)
                    nmi_ensemble.append(nmi)
        return
        df = pd.DataFrame({
            "DATABASE": datasets_ensemble,
            "ALGORITHM": names_ensemble,
            "TIME TAKEN": total_times_ensemble,
            "NMI": nmi_ensemble,
            "ACC": acc_ensemble,
        })

        print(f'{path}{timenow}_ReducedSS_{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')
        df.to_csv(f'{path}{timenow}_ReducedSS_{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

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

        print(f'{path}{timenow}_ALLACC_ReducedSS_epoch{epoch}_s{sample_size}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')
        df.to_csv(f'{path}{timenow}_ALLACC_ReducedSS_epoch{epoch}_s{sample_size}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

if __name__ == "__main__":

    sample_size = 2000
    batch_size = 1000
    lr_G = [0.007]
    lrates = [0.001, 0.0001, 0.00001]
    lratess = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    lratesss = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    lratessss = [0.00000003, 0.00000004, 0.00000005, 0.00000006, 0.00000007, 0.00000008, 0.00000009]
    lr_D = 0.001

    seeds = [100]#[1, 42, 3, 4, 5] #-> 1 does not work?

    for lr in lr_G:
        for seed in seeds:
            torch.manual_seed(seed)   #first 42
            np.random.seed(seed)
            #full_space(sample_size, batch_size, lr, lr_D, 2000, seed)
            run_experiment(sample_size, batch_size, lr, lr_D, 2000, seed)