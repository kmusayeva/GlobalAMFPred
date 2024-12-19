import numpy as np
from mlp.multi_label_propagation import *


class HarmonicFunction:
    def __init__(self, sigma: float, nn: int, iters: int = None):
        """
        Harmonic Function Model for label propagation.

        Args:
            sigma (float): Length-scale parameter of Gaussian kernel.
            nn (int): Number of neighbors.
            iters (int, optional): Number of iterations for iterative label propagation.
        """
        self.sigma = sigma
        self.nn = nn
        self.iters = iters
        self.X = None
        self.Y_train = None
        self.train_indices = None


    def fit(self, X: np.ndarray, Y_train: np.ndarray, train_indices: np.ndarray):
        """
        Args:
            dist_matrix_squared (np.ndarray): Precomputed Euclidean distance matrix.
            Y_train (np.ndarray): Training labels or response matrix.
            train_indices (np.ndarray): Indices of training data points.
        """
        
        #if self.dist_matrix_squared is None or self.Y_train is None or self.train_indices is None:
        #    raise ValueError("The model is not fitted yet. Call 'fit' with training data first.")

        self.X = X
        self.Y_train = Y_train
        self.train_indices = train_indices
        



    def predict(self, dist_matrix_squared: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Args:
            test_indices (np.ndarray): Indices of test data points.

        Returns:
            np.ndarray: Predicted soft labels for test data points.
        """
        
        transition_matrix = transition_matrix_gaussian(
            dist_matrix_squared,
            label=self.train_indices,
            sigma=self.sigma,
            nn=self.nn,
            row_norm=False
                    )

        P_UL = transition_matrix[np.ix_(test_indices, self.train_indices)]
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
