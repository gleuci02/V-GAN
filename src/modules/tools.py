import numpy as np
import pandas as pd
import random
from torch import nn, optim

def aggregator_funct(decision_function: np.array, type: str = "avg", weights: np.ndarray = None) -> np.ndarray:
    assert type in ["avg", "exact"], f"{type} aggregation not found"

    if type == "avg":
        return np.average(decision_function, axis=1, weights=weights)

    if type == "exact":
        weights = weights/weights.sum()
        random_indexes = random.choices(
            range(decision_function.shape[1]), k=decision_function.shape[0])
        aggregated_scores = [weights[random_indexes[i]]
                             * (decision_function[i])[random_indexes[i]] for i in range(decision_function.shape[0])]
        return aggregated_scores


def numeric_to_boolean(num_array, n_features):
    res = []
    for array in num_array:
        bool_array = [False for i in range(n_features)]
        for entry in array:
            bool_array[entry] = True
        res.append(bool_array)

    return res

def reduce_subspaces(u, proba):

    sorted_vectors = sorted(u.tolist(), key=lambda v: sum(v), reverse=True)

    result = u.tolist()

    for i, v in enumerate(u.tolist()):

        for j, v2 in enumerate(sorted_vectors[i:]):

            result_vector = [a or b for a, b in zip(v, v2)]
            
            if result_vector in sorted_vectors:
            
                    sorted_vectors.remove(v2)
                    result.remove(v2)
                    proba[i] += proba[j]
                    proba = np.delete(proba, j)

    return result, proba


def pretrain_autoencoder(detector, dataloader, epochs=10, lr=0.001, device='cuda'):
    detector.to(device)
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.view(batch.size(0), -1).to(device)
            optimizer.zero_grad()
            _, batch_dec = detector(batch)
            loss = loss_fn(batch_dec, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return detector