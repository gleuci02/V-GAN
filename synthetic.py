from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
from sklearn import cluster
from src.cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from src.modules.tools import reduce_subspaces
import seaborn as sns
import torch
from src.modules.od_module import VGAN
import numpy as np
import datetime
from pathlib import Path
from metrics import normalized_mutual_info_score, clustering_accuracy

ALGORITHMS = {
    #"kmeans": cluster.KMeans(n_clusters=3), #mini batch kmeans?
    #"SSC_OMP": SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5),
    #"Elastic": ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9),
    "Spectral_clustering": cluster.SpectralClustering(n_clusters=3, affinity='nearest_neighbors', n_neighbors=5)
}

def generate_clustering_ensemble(clusterings):
    
    n_samples = clusterings.shape[1]
    co_matrix = np.zeros((n_samples, n_samples))
    
    for clustering in clusterings:
        for i in range(n_samples):
            for j in range(i, n_samples):
                if clustering[i] == clustering[j] and i == j:
                    co_matrix[i, j] += 1
                elif clustering[i] == clustering[j]:
                    co_matrix[i, j] += 1
                    co_matrix[j, i] += 1

    co_matrix /= co_matrix.diagonal()  # Normalize by number of clusterings

    spectral = cluster.SpectralClustering(3, affinity="precomputed")
    final_clusters = spectral.fit_predict(co_matrix)

    return final_clusters

def vgan_training(vgan, X_train):

    vgan.fit(X_train)

    vgan.approx_subspace_dist(500, add_leftover_features=True)

    subspaces = vgan.subspaces

    proba = vgan.proba

    print(len(subspaces))

    subspaces, proba = reduce_subspaces(subspaces, proba)

    print(len(subspaces))

    tup = zip(subspaces, proba)

    subspaces = sorted(tup, key=lambda tup: tup[1], reverse=True)[:30]

    subspaces = [x[0] for x in subspaces]

    return torch.tensor(subspaces).cpu()


def run_experiment(batch_size, lr_G, lr_Ds, epoch):

    vgan = VGAN(epochs = epoch, temperature=10, batch_size=500, path_to_directory=Path()/ "experiments" / f"Example_dataset_{datetime.datetime.now()}", iternum_d=1, iternum_g=5,lr_G = lr_G, lr_D = lr_Ds)

    #np.random.seed(214)
    np.random.seed(vgan.seed)
    X_moons, _ = make_moons(n_samples=350, noise=0.1, random_state=214)
    X_blobs, _ = make_blobs(n_samples=50, centers=[[-0.5,-0.5]], cluster_std=[0.1], random_state=214)
    X_noise = np.random.uniform(low=-1.5, high=2.0, size=(100,2))

    X_vectors = np.vstack([X_moons, X_blobs, X_noise])

    fig, ax = plt.subplots(figsize=(6,6))
    plot = sns.scatterplot(x=X_vectors[:,0], y=X_vectors[:,1], ax=ax)
    fig = plot.get_figure()
    fig.savefig("scatterplot.png")

    X_vectors = torch.Tensor(X_vectors)

    #------Preprosessing with VGAN-----#
    subspaces = vgan_training(vgan, X_vectors)
    #------End of Preprosessing with VGAN-----#

    for name in ALGORITHMS:

        clusterings = []
        algorithm = ALGORITHMS[name]

        for subspace in subspaces:

            X_new = X_vectors * subspace

            y_pred = algorithm.fit_predict(X_new)

            print(y_pred)

            clusterings.append(y_pred)

        clusterings = np.array(clusterings)
        final_cluster = generate_clustering_ensemble(clusterings)

        fig, ax = plt.subplots( figsize=(12,6), ncols=2 )

        #final_cluster = algorithm.fit_predict(X_vectors)

        plot = sns.scatterplot(
            x=X_vectors[:,0], 
            y=X_vectors[:,1], 
            hue=final_cluster, 
            palette='tab20',
            ax=ax[0]
        )

        ax[0].legend(loc='upper right', title='Cluster')

        ax[0].set_title(f"K-Means Ensemble)")

        fig = plot.get_figure()
        fig.savefig("SC_ensemble.png")


if __name__ == "__main__":

    batch_size = 1000
    lr_G = 0.01
    lr_D = 0.01

    run_experiment(batch_size, lr_G, lr_D, 150)

