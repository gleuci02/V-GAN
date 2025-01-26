from datasets.mnist import load_mnist, load_fashion_mnist
#from datasets.cifar import load_cifar10, load_cifar100
#from datasets.stl import load_stl10
#from algorithms.EDSC import edesc
from sklearn import cluster
#from src.cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from metrics import normalized_mutual_info_score, clustering_accuracy
import pandas as pd
#from src.modules.tools import reduce_subspaces
from pathlib import Path
from torch.utils.data import DataLoader
from src.modules.od_module import VGAN
import matplotlib.pyplot as plt
import numpy as np
import torch
import datetime
import time
import torch_two_sample as tts


ALGORITHMS = {
    "kmeans": cluster.KMeans(n_clusters=10),
}

DATASETS = {
    "MNIST": load_mnist,
}

def run_experiment(batch_size, lr_G, lr_Ds, epoch):
       
        datasets = []
        names = []
        total_times = []
        nmis = []
        accuracys = []    # 5 datasets


        # Load dataset
        for i, dataset in enumerate(DATASETS):
                vgan = VGAN(epochs = epoch, temperature=10, batch_size=batch_size, path_to_directory=Path()/ "experiments" / f"Example_dataset_{datetime.datetime.now()}", iternum_d=1, iternum_g=5,lr_G = lr_G, lr_D = lr_D)

                print(f"CURRENTLY WORKING FOR {dataset}")

                dataloader = DataLoader(DATASETS[dataset](), batch_size=batch_size, shuffle=False)

                X, y_true = next(iter(dataloader))

                #------Preprosessing with VGAN-----#
                vgan.fit(X)

                vgan.approx_subspace_dist(batch_size, add_leftover_features=True)
                #------End of Preprosessing with VGAN-----#

                for name in ALGORITHMS:
                    starting_time = time.time()
                    # Initialize algorithm
                    algorithm = ALGORITHMS[name]

                    subspaces = vgan.subspaces

                    proba = vgan.proba

                    np.random.seed(vgan.seed)

                    #Take random sample with a given distribution
                    subspace_index = np.random.choice(list(range(0, 101)), p=proba)

                    subspace = subspaces[subspace_index]

                    X = [x[subspace] for x in X]

                    y_pred = algorithm.fit_predict(X)

                    nmi = normalized_mutual_info_score(y_true, y_pred)
                    acc = clustering_accuracy(y_true, y_pred)
                    
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

        plt.close()
        plt.plot(vgan.pvalues)
        plt.savefig("Kmeans_pvalues.png", dpi=300)
        plt.close()

        print(df)
        df.to_csv(f'Smaller_NN_epoch{epoch}_b{batch_size}_lr_G{lr_G}lr_D{lr_D}.csv')

if __name__ == "__main__":

    batch_size = 100
    lr_G = 0.01
    lr_D = 0.01

    run_experiment(batch_size, lr_G, lr_D, 150)