from typing import Tuple
from collections import deque

import numpy as np
from scipy.spatial.distance import pdist, squareform

class SNNDPC:
    def __init__(self, k: int, n_clusters: int):
        self.k = k
        self.nc = n_clusters

    def fit_predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unassigned = -1
        n, _ = data.shape
        k = self.k
        nc = self.nc

        # 1) Distance matrix
        D = squareform(pdist(data))

        # 2) k-NN indices and neighbor-sets
        idx_sort = np.argsort(D, axis=1)
        knn = idx_sort[:, :k]
        neighbors = [set(knn[i]) for i in range(n)]

        # 3) Similarity matrix (on the fly shared-neighbor)
        sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                shared = neighbors[i] & neighbors[j]
                if i in shared and j in shared:
                    shared_list = list(shared)
                    dist_sum = D[i, shared_list].sum() + D[j, shared_list].sum()
                    cnt = len(shared)
                    sim[i, j] = sim[j, i] = cnt * cnt / dist_sum

        # 4) ρ = sum of top-k similarities per point
        rho = np.sort(sim, axis=1)[:, -k:].sum(axis=1)

        # 5) δ
        #    pre-compute sum of distances to k neighbors
        knn_dist_sum = D[np.arange(n)[:,None], knn].sum(axis=1)
        idx_rho_desc = np.argsort(rho)[::-1]
        delta = np.full(n, np.inf)
        # for all but the very highest-ρ point
        for i, a in enumerate(idx_rho_desc[1:], 1):
            better = idx_rho_desc[:i]
            # weight by neighbor-distance sums
            vals = D[a, better] * (knn_dist_sum[a] + knn_dist_sum[better])
            delta[a] = vals.min()
        # for the global max-ρ
        delta[idx_rho_desc[0]] = delta.max()

        # 6) γ
        gamma = rho * delta

        # 7) select centroids
        centroids = np.sort(np.argsort(gamma)[-nc:])
        labels = np.full(n, unassigned, int)
        labels[centroids] = np.arange(nc)

        # 8) assign non-centroids: BFS over the k-NN graph
        queue = deque(centroids.tolist())
        while queue:
            a = queue.popleft()
            for b in knn[a]:
                if labels[b] == unassigned:
                    # shared-neighbor count on the fly
                    if len(neighbors[a] & neighbors[b]) >= k / 2:
                        labels[b] = labels[a]
                        queue.append(b)

        # 9) any still unassigned? do majority-vote among their k nearest
        unassigned_idx = np.where(labels == unassigned)[0]
        while unassigned_idx.size:
            counts = np.zeros((unassigned_idx.size, nc), int)
            for i, a in enumerate(unassigned_idx):
                for b in idx_sort[a, :k]:
                    lb = labels[b]
                    if lb != unassigned:
                        counts[i, lb] += 1
            max_count = counts.max()
            if max_count > 0:
                # assign all with a clear majority
                i_flat, c_flat = np.where(counts == max_count)
                for i_row, cluster in zip(i_flat, c_flat):
                    labels[unassigned_idx[i_row]] = cluster
                unassigned_idx = np.where(labels == unassigned)[0]
            else:
                # no decisive neighbor: bail out by increasing k
                k += 1

        return labels
        return centroids, labels

    def set_params(self, n_clusters: int):
        self.nc = n_clusters

    def get_params(self):
        return {"n_clusters": self.nc, "k": self.k}