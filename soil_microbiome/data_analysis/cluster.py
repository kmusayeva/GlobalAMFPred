"""
Implements k-means clustering.
Author: Khadija Musayeva
Email: khmusayeva@gmail.com
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


def kmeans_cluster(X: np.ndarray):
    range_n_clusters = list(range(1, 11))
    inertias = []  # Inertia is the sum of squared distances of samples to their closest cluster center
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        # Silhouette score can only be calculated where n_clusters > 1
        if n_clusters > 1:
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        else:
            silhouette_scores.append(None)

    # Plotting the Elbow Method
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range_n_clusters, inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    # Plotting the Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(range_n_clusters[1:], silhouette_scores[1:], marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()
