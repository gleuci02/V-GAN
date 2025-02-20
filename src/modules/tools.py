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

    for i, v in enumerate(u.tolist()):

        for j, v2 in enumerate(sorted_vectors[i+1:]):

            result_vector = [a or b for a, b in zip(v, v2)]
            
            if result_vector in sorted_vectors:
            
                    sorted_vectors.remove(v2)
                    result.remove(v2)
                    proba[i] += proba[j]
                    proba = np.delete(proba, j)

    return result, proba