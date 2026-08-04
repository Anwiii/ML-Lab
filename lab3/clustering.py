# clustering.py
# A11 — K-Means from scratch

import numpy as np
from distances import minkowski_distance
from stats_funcs import compute_mean


def assign_clusters(data, centroids):
    """
    Assign each point to the nearest centroid using Euclidean distance.
    Returns array of cluster indices.
    """
    labels = []
    for point in data:
        dists = [minkowski_distance(point, c, p=2) for c in centroids]
        labels.append(np.argmin(dists))
    return np.array(labels)


def recompute_centroids(data, labels, k):
    """
    Recompute each centroid as the mean of its assigned points.
    If a cluster is empty, reinitialize to a random data point.
    """
    data = np.array(data, dtype=float)
    centroids = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centroids.append(compute_mean(cluster_points))
        else:
            # Reinitialize empty cluster
            centroids.append(data[np.random.randint(0, len(data))])
    return np.array(centroids)


def kmeans(data, k, max_iter=100, random_state=42):
    """
    K-Means clustering from scratch (Algorithm 8.1, Tan et al.).

    Steps:
    1. Select K random points as initial centroids.
    2. Assign each point to its nearest centroid.
    3. Recompute centroids as mean of assigned points.
    4. Repeat 2-3 until centroids do not change.

    Returns (labels, centroids, iterations_taken).
    """
    np.random.seed(random_state)
    data = np.array(data, dtype=float)

    # Step 1: Random initialization
    idx = np.random.choice(len(data), k, replace=False)
    centroids = data[idx]

    for iteration in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = recompute_centroids(data, labels, k)

        # Convergence check
        if np.allclose(centroids, new_centroids):
            return labels, new_centroids, iteration + 1

        centroids = new_centroids

    return labels, centroids, max_iter


def compute_wcss(data, labels, centroids):
    """
    Compute Within-Cluster Sum of Squared Errors (WCSS).
    Lower WCSS = tighter, more compact clusters.
    """
    data = np.array(data, dtype=float)
    wcss = 0.0
    for i, point in enumerate(data):
        wcss += np.sum((point - centroids[labels[i]]) ** 2)
    return wcss


def normalize(data):
    """
    Min-Max normalize data to [0, 1] column-wise.
    Avoids division by zero with a small epsilon.
    """
    data = np.array(data, dtype=float)
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    return (data - col_min) / (col_max - col_min + 1e-9)
