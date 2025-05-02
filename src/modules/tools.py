import numpy as np
import pandas as pd
import random


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
    indices_to_remove = set()

    for i, v in enumerate(result):
        for j in range(i, len(sorted_vectors)):  # Iterate over indices instead
            if j in indices_to_remove:
                continue  # Skip already removed indices

            v2 = sorted_vectors[j]
            result_vector = [a or b for a, b in zip(v, v2)]

            if result_vector in sorted_vectors:
                indices_to_remove.add(j)
                proba[i] += proba[j]  # Accumulate probability

    # Remove elements after iteration
    sorted_vectors = [vec for idx, vec in enumerate(sorted_vectors) if idx not in indices_to_remove]
    result = [vec for idx, vec in enumerate(result) if idx not in indices_to_remove]
    proba = np.delete(proba, list(indices_to_remove))

    return result, proba