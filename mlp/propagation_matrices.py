"""
Creates propagation matrices.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
import numpy as np
import pandas as pd
import scipy as sc
import osqp
from sklearn.preprocessing import MinMaxScaler
from sklearn .preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances


def Laplacian_matrix(dist_matrix_squared: np.ndarray, sigma: float):
    """
    Computes Laplacian matrix.
    :param dist_matrix_squared: squared Euclidean distance matrix
    :param sigma: length-scale parameter of the Gaussian kernel
    :return: np.ndarray, symmetric normalized Laplacian
    """
    W = similarity_matrix(dist_matrix_squared, sigma)
    W_row_sum = W.sum(axis=1)
    D = np.diag(W_row_sum)
    return np.subtract(D, W)


def propagation_matrix_normalized_Laplacian(dist_matrix_squared: np.ndarray, sigma: float):
    """
    Computes normalized Laplacian matrix.
    :param dist_matrix_squared: squared Euclidean distance matrix
    :param sigma: length-scale parameter of the Gaussian kernel
    :return: np.ndarray, symmetric normalizaed Laplacian
    """
    P = similarity_matrix(dist_matrix_squared, sigma)
    P_row_sum = P.sum(axis=1)
    P_diag = np.diag(P_row_sum)
    D = sc.linalg.fractional_matrix_power(P_diag, -0.5)
    return (D.dot(P).dot(D)).round(3)


def transition_matrix_gaussian(dist_matrix_squared, label, sigma, nn, row_norm=False):
    """
    Computes transition matrix based on the Gaussian kernel.
    :param dist_matrix_squared:
    :param label:
    :param sigma:
    :param nn:
    :param row_norm:
    :return:
    """
    P = similarity_matrix(dist_matrix_squared, sigma, diagonal=True)
    sorted_indices = np.argsort(-P, axis=1)  # Sort indices in descending order
    P_prime = np.zeros_like(P)
    for i in range(P.shape[0]):
        top_n_indices = sorted_indices[i, :nn]  # Get indices of top n neighbors
        P_prime[i, top_n_indices] = P[i, top_n_indices]  # Keep only top n neighbors

    P_prime[label, :] = 0
    np.fill_diagonal(P_prime, 1)
    # Z = np.delete(P_prime, label, axis=0)
    # print(Z)
    if row_norm:
        return row_normalize(P_prime)
    return column_row_normalize(P_prime)


def transition_matrix_linear_patch(X, dist_matrix, nn):
    """
    implements the propagation matrix of Wang and Zang, 2008
    :param X:
    :param dist_matrix:
    :param nn:
    :return:
    """
    nn_matrix = np.argsort(dist_matrix)
    n = X.shape[0]
    q = np.zeros(nn)
    A = sc.sparse.csc_matrix(np.concatenate((np.array([[1] * nn]), np.eye(nn)), axis=0))
    l = np.concatenate((np.array([1]), np.zeros(nn)))
    u = np.ones(nn + 1)
    W = np.zeros((n, n))
    for i in range(n):
        neighbours = nn_matrix[i, 1:nn + 1]
        X_neighbours = X[neighbours, :]
        # X_neighbours_matrix = X_neighbours
        Z = X_neighbours - X[i, :]
        G = np.matmul(Z, Z.T)
        P = sc.sparse.csc_matrix(2 * G)
        prob = osqp.OSQP()
        # Setup workspace and change alpha parameter
        prob.setup(P, q, A, l, u, alpha=1.0, verbose=False, eps_abs=1e-10, eps_rel=1e-10, eps_prim_inf=1e-10,
                   eps_dual_inf=1e-10)

        # Solve problem
        res = prob.solve()
        W[i, neighbours] = res.x
    return W


def chi_square_transform(Y):
    """
    Computes the chi-square transformation
    :param Y:
    :return:
    """
    total_sum = Y.values.sum()
    row_sum = Y.sum(axis=1)
    col_sum = Y.sum(axis=0)
    step1 = Y.div(row_sum, axis=0)
    step2 = step1 / np.sqrt(col_sum)
    Y_chi_square = np.sqrt(total_sum) * step2
    return Y_chi_square


def dist_matrix(X):
    """
    computes euclidean distance
    :param X:
    :return:
    """
    normalized_X = StandardScaler().fit_transform(X)
    distance_matrix = pairwise_distances(normalized_X, metric='euclidean')
    return distance_matrix


def similarity_matrix(dist_matrix_squared, sigma, diagonal=False):
    """
    computes the similarity matrix
    :param dist_matrix_squared:
    :param sigma:
    :param diagonal:
    :return:
    """
    W = np.exp(-dist_matrix_squared / sigma ** 2)
    if diagonal is False:
        np.fill_diagonal(W, 0)
    return W


def row_normalize(A):
    """
    row-normalize the given matrix
    :param A:
    :return:
    """
    row_sum = np.sum(A, axis=1, keepdims=True)
    A_row_norm = A / row_sum
    return A_row_norm


def column_row_normalize(A):
    """
    column-normalize the given matrix
    :param A:
    :return:
    """
    A_col_norm = A / A.sum(axis=0)
    return row_normalize(A_col_norm)


def convert_to_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    else:
        return pd.DataFrame(x)

