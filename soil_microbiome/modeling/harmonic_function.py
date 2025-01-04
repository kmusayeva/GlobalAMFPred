"""
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""

import numpy as np
from mlp.multi_label_propagation import *


class HarmonicFunction:
    def __init__(self, sigma: float, nn: int, iters: int = None):
        """
        Harmonic Function for label propagation.

        @param: sigma: Length-scale parameter of Gaussian kernel.
        @param: nn: Number of neighbors.
        @param: iters (optional): Number of iterations for iterative label propagation.
        """
        self.sigma = sigma
        self.nn = nn
        self.iters = iters
        self.X = None
        self.Y_train = None

    def fit(self, X: np.ndarray, Y_train: np.ndarray):
        """
        @param: dist_matrix_squared (np.ndarray): Precomputed Euclidean distance matrix.
        @param: Y_train (np.ndarray): Training labels or response matrix.
        """
        self.X = X
        self.Y_train = Y_train

    def predict(
        self,
        dist_matrix_squared: np.ndarray,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> np.ndarray:
        """
        @param dist_matrix_squared: squared distance matrix of the whole input matrix
        @param train_indices: indices of the training data
        @param test_indices: indices of the test data
        @return: a matrix of soft labels
        """

        transition_matrix = transition_matrix_gaussian(
            dist_matrix_squared,
            label=train_indices,
            sigma=self.sigma,
            nn=self.nn,
            row_norm=False,
        )

        P_UL = transition_matrix[np.ix_(test_indices, train_indices)]
        P_UU = transition_matrix[np.ix_(test_indices, test_indices)]

        u = len(test_indices)
        I = np.eye(len(P_UU))

        if self.iters is None:
            A = I - P_UU
            np.fill_diagonal(A, A.diagonal() + 1e-15)
            B = np.linalg.pinv(A)
            soft_labels = B.dot(P_UL).dot(self.Y_train)
        else:
            soft_labels = np.zeros((u, self.Y_train.shape[1]))
            for _ in range(self.iters):
                soft_labels = P_UL.dot(self.Y_train) + P_UU.dot(soft_labels)

        return soft_labels.round(3)
