"""
Implements label-propagation methods.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
from .propagation_matrices import *
from .thresholding import *


def hf(dist_matrix_squared: np.ndarray, Y_train: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray, nn: int,
       sigma: float, iters=None):
    """
    Harmonic function and iterative label propagation
    :param dist_matrix_squared: Euclidean distance matrix of input data
    :param Y: response matrix
    :param train_indices: training indices
    :param test_indices: test indices
    :param sigma: length-scale parameter of Gaussian kernel
    :param nn: number of neighbours
    :param iters: number of iterations for iterative label propagation
    :return: np.ndarray, of soft labels
    """
    transition_matrix = transition_matrix_gaussian(dist_matrix_squared, label=train_indices, sigma=sigma, nn=nn,
                                                   row_norm=False)
    u = len(test_indices)
    P_UL = transition_matrix[np.ix_(test_indices, train_indices)]
    P_UU = transition_matrix[np.ix_(test_indices, test_indices)]
    I = np.eye(len(P_UU))
    if iters is None:
        A = I - P_UU
        np.fill_diagonal(A, A.diagonal() + 1e-15)
        B = np.linalg.pinv(A)
        soft_labels = B.dot(P_UL).dot(Y_train)

    else:  # if the iter parameter is not null, do iterative label propagation
        soft_labels = np.zeros((u, Y_train.shape[1]))
        for i in range(iters):
            soft_labels = P_UL.dot(Y_train) + P_UU.dot(soft_labels)

    return soft_labels.round(3)


def cm(dist_matrix_squared: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, sigma: float, reg: float):
    """
    Implements consistency method (CM) of Zhou et al., 2004
    :param dist_matrix_squared: Euclidean distance matrix of input data
    :param Y: response matrix
    :param train_indices:
    :param sigma: length-scale parameter of the gaussian kernel
    :param reg: regularization parameter in (0,1)
    :return: np.ndarray, of soft labels
    """
    transition_matrix = propagation_matrix_normalized_Laplacian(dist_matrix_squared, sigma)
    soft_labels = regularized_approach(Y.copy(), train_indices, transition_matrix, reg)
    return soft_labels


def lln(X: np.ndarray, dist_matrix_squared: np.ndarray, Y: np.ndarray, train_indices: np.ndarray, nn: int, reg: float):
    """
    locally linear neighbourhood (LLN) of Wang and Zang, 2008
    :param dist_matrix_squared: Euclidean distance matrix of input data
    :param Y: response matrix
    :param label: label matrix
    :param nn: number of neighbours
    :return: matrix of soft labels
    """
    transition_matrix = transition_matrix_linear_patch(X, dist_matrix_squared, nn)
    Y_prime = Y.copy()
    Y_prime[Y_prime == 0] = -1
    return regularized_approach(Y_prime, train_indices, transition_matrix, reg)


def regularized_approach(Y: np.ndarray, label: np.ndarray, transition_matrix: np.ndarray, reg: float):
    """
    common part to the regularized approaches, CM and LLN
    :param Y: response matrix
    :param label: label matrix
    :param transition_matrix
    :param reg: regularization parameter in (0,1)
    :return: matrix of soft labels
    """
    n = Y.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[label] = False
    Y[mask, :] = 0
    I = np.eye(n)
    A = np.linalg.inv(I - reg * transition_matrix)
    preds = A.dot(Y)
    preds = preds[mask, :]
    return preds
