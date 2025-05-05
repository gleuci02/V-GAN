from datasets.mnist import load_mnist, load_fashion_mnist
from datasets.cifar import load_cifar10, load_cifar100
from datasets.stl import load_stl10
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
import numpy as np
import torch
import datetime
import time
from sklearn.ensemble import BaggingClassifier
from src.modules.tools import pretrain_autoencoder, train_vae
from src.modules.network_module import Detector, Encoder, Decoder

path = "results/results_latentSpace/"

ALGORITHMS = {
    "kmeans": cluster.KMeans(n_clusters=10), #mini batch kmeans?
    "SSC_OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    "Elastic": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    "Spectral_clustering": cluster.SpectralClustering(n_clusters=10, affinity='nearest_neighbors', n_neighbors=5)
}

DATASETS = {
    #"CALTECH101": load_caltech101,
    #"MNIST": load_mnist,
    #"CIFAR10": load_cifar10,
    #"STL10": load_stl10,
    #"CIFAR100": load_cifar100,
    "FASHION_MNIST": load_fashion_mnist,
}

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
    plt.savefig('decoded.png')

def plot_subspaces(vgan, images, U, dataset, shape):
    # Select 20 sample images
    num_images = 20
    rows, cols = 4, 5  # 4 rows, 5 columns
    
    U = torch.unflatten(U, 1, shape[1:])

    enc = vgan.detector.encoder
    dec = vgan.detector.decoder

    for i in range(num_images):
        col_idx = i % cols
        if col_idx < len(U):
            proj_enc = images.to('cuda') * U[col_idx].to('cuda')
            images[i] = proj_enc[i]

    # Reshape images for plotting
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy().astype("float32")
    
    # Create subplots
    fig, axes = plt.subplots(rows + 1, cols, figsize=(10, 10))
    
    #U = dec(U.float().to('cuda')).permute(0, 2, 3, 1).cpu().detach().numpy()

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
    #latent_size = max(int(X_train.flatten(1, -1).shape[1]/16), 1)
    #vgan.load_models("./experiments/Example_normal_2025-03-16 14:17:39.055543_vmmd/models/generator_0.pt", X_train.shape[2], latent_size)
    vgan.approx_subspace_dist(500, add_leftover_features=True)

    subspaces = vgan.subspaces

    proba = vgan.proba

    subspaces, proba = reduce_subspaces(subspaces, proba)

    tup = zip(subspaces, proba)

    subspaces = sorted(tup, key=lambda tup: tup[1], reverse=True)[:30]

    subspaces = [x[0] for x in subspaces]

    return torch.tensor(subspaces).cpu()


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
                #vgan = VGAN(epochs = epoch, temperature=10, batch_size=batch_size, path_to_directory=Path()/ "experiments" / f"Example_dataset_{datetime.datetime.now()}", iternum_d=1, iternum_g=5,lr_G = lr_G, lr_D = lr_Ds)
                vgan = VMMD(epochs=epoch, path_to_directory=Path() / "experiments" /f"Example_normal_{datetime.datetime.now()}_vmmd", lr=lr_G)
                vgan.seed = 42
                torch.manual_seed(42)
                np.random.seed(42)
                vgan.dataset = dataset
                print(f"CURRENTLY WORKING FOR {dataset}")

                dataset_train, dataset_test = DATASETS[dataset]()

                # Get indices where the label is 1 (Pants)
                train_indices = [i for i, (_, label) in enumerate(dataset_train) if label == 1] #17
                test_indices = [i for i, (_, label) in enumerate(dataset_test) if label == 1]

                # Create filtered datasets
                filtered_train = Subset(dataset_train, train_indices)
                filtered_test = Subset(dataset_test, test_indices)

                dataloader_train = DataLoader(filtered_train, batch_size=sample_size, shuffle=False)

                #dataloader_test = DataLoader(dataset_test, batch_size=int(sample_size / 10), shuffle=False)

                X_train, y_train = next(iter(dataloader_train))
                #X_test, y_test = next(iter(dataloader_test))
                
                latent_size = 128#max(int(X_train.flatten(1, -1).shape[1]/16), 1) #X_train.flatten(1, -1).shape[1] #
                ndims = X_train.shape[2]
                channel = X_train.shape[1]

                #autoencoder = Detector(latent_size, ndims, channel, Encoder, Decoder)
                #autoencoder.to('cuda')
                #autoencoder = pretrain_autoencoder(autoencoder, dataloader_train, epochs=50, lr=0.002)

                #torch.save(autoencoder.encoder.state_dict(), f"./AE_Weights/encoder_weights_{dataset}.pth")
                #torch.save(autoencoder.decoder.state_dict(), f"./AE_Weights/decoder_weights_{dataset}.pth")
                
                #autoencoder.encoder.load_state_dict(torch.load(f"./AE_Weights/encoder_weights_{dataset}.pth"))
                #autoencoder.decoder.load_state_dict(torch.load(f"./AE_Weights/decoder_weights_{dataset}.pth"))

                # Call visualization function

                #visualize_reconstruction(autoencoder, dataloader_train)
                
                shape = X_train.shape

                #------Preprosessing with VGAN-----#
                subspaces = vgan_training(vgan, X_train)
                print(shape)

                #np.random.seed(42)
                #n_features = X_train.shape[1]
                #n_models = 3  # Number of classifiers
                #feature_masks = [np.random.choice([0, 1], size=n_features, p=[0.5, 0.5]) for _ in range(n_models)]

                plot_subspaces(vgan, X_train, subspaces, dataset, shape[1:])
                #test = vgan.check_if_myopic(X_train.detach().numpy(), [vgan.bandwidth.cpu()], len(X_train))
                #test.to_csv(f'{path}ifmyopic{dataset}.csv')
                #print(f'{path}ifmyopic{dataset}.csv')


                plt.clf()
                plt.figure(figsize=(20, 6))
                plt.plot(vgan.train_history["generator_loss"])
                plt.savefig(f'{path}VGAN_GLOSS_{dataset}_ReducedSS_Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D.png', dpi=300)
                exit()
                #continue
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

                        subspace = torch.unflatten(subspace, 0, shape[2:])
                        X_new = X_train * subspace
                        X_new = torch.flatten(X_new, 1, -1)

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

                    #bandwiths.append(bandwith)
                    #results.append(result)
                    datasets_ensemble.append(dataset)
                    names_ensemble.append(name)
                    total_times_ensemble.append(total_time)
                    acc_ensemble.append(acc)
                    nmi_ensemble.append(nmi)

                    plt.clf()
                    plt.figure(figsize=(20, 6))
                    plt.plot(vgan.train_history["generator_loss"])
                    plt.show()
                    plt.savefig(f'{path}VGAN_GLOSS_{dataset}_ReducedSS_Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D.png', dpi=300)
                    plt.clf()
        
        df = pd.DataFrame({
            "DATABASE": datasets_ensemble,
            "ALGORITHM": names_ensemble,
            "TIME TAKEN": total_times_ensemble,
            "NMI": nmi_ensemble,
            "ACC": acc_ensemble,
            #"PValue": results,
            #"bandwidth": bandwiths
        })

        #df.to_csv(f'FB_Ensemble_ReducedSS_Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')
        df.to_csv(f'{path}PRETRAIN_AE_WITHCONV_ReducedSS_{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

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

        df.to_csv(f'{path}ALLACC_PRETRAIN_AE_WITHCONV_ReducedSS_epoch{epoch}_s{sample_size}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

if __name__ == "__main__":

    sample_size = 2000
    batch_size = 1000
    lr_G = 0.007
    lr_D = 0.001

    torch.manual_seed(42)
    np.random.seed(42)

    run_experiment(sample_size, batch_size, lr_G, lr_D, 2000)
