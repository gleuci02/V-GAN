import numpy as np
import pandas as pd
import random
import torch
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

#def reduce_subspaces(u, proba):
#
#    sorted_vectors = sorted(u.tolist(), key=lambda v: sum(v), reverse=True)
#
#    result = u.tolist()
#
#    for i, v in enumerate(u.tolist()):
#
#        for j, v2 in enumerate(sorted_vectors[i:]):
#
#            result_vector = [a or b for a, b in zip(v, v2)]
#
#            if result_vector in sorted_vectors:
 #                   
  #                  sorted_vectors.remove(v2)
   #                 result.remove(v2)
    #                proba[i] += proba[j]
     #               proba = np.delete(proba, j)

    #return result, proba

def reduce_subspaces(u, proba):

    print(u.shape)
    u = u.tolist()

    filtered = []
    proba_filtered = []
    i = 0
    for v in u:
        p_true = sum(1 for x in v if x) / len(v)
        p_false = 1 - p_true
        # If ≥ 95% are True OR ≥ 95% are False, skip; otherwise keep
        if p_true >= 0.95 or p_false >= 0.95:
            # exclude this vector
            print("IF STATEMENT")
            continue
        filtered.append(v)
        proba_filtered.append(proba[i])
        i += 1

    sorted_vectors = sorted(filtered, key=lambda v: sum(v), reverse=True)

    result = filtered
    indices_to_remove = set()

    for i, v in enumerate(result):
        for j in range(i + 1, len(sorted_vectors)):  # Iterate over indices instead
            if i in indices_to_remove:
                continue  # Skip already removed indices

            v2 = sorted_vectors[j]
            result_vector = [a or b for a, b in zip(v, v2)]

            if result_vector in sorted_vectors:
                indices_to_remove.add(i)
                proba_filtered[i] += proba_filtered[j]  # Accumulate probability

    # Remove elements after iteration
    sorted_vectors = [vec for idx, vec in enumerate(sorted_vectors) if idx not in indices_to_remove]
    result = [vec for idx, vec in enumerate(result) if idx not in indices_to_remove]
    proba_filtered = np.delete(proba_filtered, list(indices_to_remove))

    return result, proba_filtered


def pretrain_autoencoder(detector, dataloader, epochs=10, lr=0.001, device='cuda'):
    detector.to(device)
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            _, batch_dec = detector(batch)  # Forward pass
            shape = batch.shape
            batch_dec = torch.unflatten(batch_dec, 1, (shape[1], shape[2], shape[3]))      #(3, 32, 32))
            loss = loss_fn(batch_dec, batch)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    return detector  # Return pretrained detector

def train_vae(model, train_loader, lr=1e-3, num_epochs=20, kld_weight=0.00025, device=None):
    """
    Trains a Variational Autoencoder (VAE).
    
    :param model: The VAE model instance.
    :param dataset: The dataset to train on.
    :param latent_dim: Dimension of the latent space.
    :param batch_size: Batch size for training.
    :param lr: Learning rate for the optimizer.
    :param num_epochs: Number of training epochs.
    :param kld_weight: Weight for KL divergence term.
    :param device: The device (CPU or CUDA) to run training on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _, mu, log_var = model(data)
            
            # Compute loss
            loss_dict = model.loss_function(outputs, data, mu, log_var, M_N=kld_weight)
            loss = loss_dict['loss']
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    return model