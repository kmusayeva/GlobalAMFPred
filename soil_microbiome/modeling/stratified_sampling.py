"""
Implements stratified sampling strategy of
Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the stratification of multi-label data.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union


def iterative_stratification(Y: np.ndarray, k: int, proportions: List[float]) -> List[np.ndarray]:
    """
    Perform stratified sampling based on the response matrix Y, returning k disjoint subsets of indices.
    @param Y: A binary response matrix of shape (n_samples, n_labels).
    @param k: Desired number of subsets.
    @return: A list of k disjoint subsets of indices.
    """
    n_samples, n_labels = Y.shape
    D = list(range(n_samples))  # List of all sample indices
    subsets = [[] for _ in range(k)]  # Initialize empty subsets for indices

    # Calculate the desired count of samples in each subset
    c = [round(n_samples * prop) for prop in proportions]

    if sum(c) < n_samples:
        c[0] += n_samples - sum(c)

    undistributed_labels = set(range(n_labels))
    while undistributed_labels:
        # Dynamically determine label order based on current distribution of undistributed labels
        label_frequencies = {label: Y[list(D), label].sum() for label in undistributed_labels}
        # Order labels by their frequency, rare ones first
        label_order = sorted(label_frequencies, key=label_frequencies.get)
        i = label_order[0]  # For each label, distribute its indices across the subsets
        Di = [idx for idx in D if Y[idx, i] == 1]  # Indices with the i-th label
        cij = [round(len(Di) * prop) for prop in proportions]  # Desired count per subset for this label
        for idx in Di:
            # Choose the subset for this index
            subset_sizes = [len(subsets[j]) for j in range(k)]
            subset_deficits = [c[j] - subset_sizes[j] for j in range(k)]  # Remaining capacity in each subset
            # Prioritize subsets with the largest deficit, then by their original capacity
            priorities = sorted(range(k), key=lambda x: (-cij[x], -subset_deficits[x]))
            # print(f"priorities: {priorities}")
            chosen_subset = priorities[0]
            subsets[chosen_subset].append(idx)
            D.remove(idx)  # Remove the index from D to ensure it's not used again
            # Update the counts
            for j in range(k):
                cij[j] -= 1 if j == chosen_subset else 0
        undistributed_labels.remove(i)
    if len(D) > 0:
        subsets[0] = [*subsets[0], *D]

    return subsets
