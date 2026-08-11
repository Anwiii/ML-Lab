# A11 - K-Means Clustering, built from our own A4 (distance) and A8 (mean) functions

import pandas as pd
import random
from four import minkowski_distance
from eight import calculate_mean


def assign_points(data, centroids):
    labels = []
    for point in data:
        distances = []
        for centroid in centroids:
            d = minkowski_distance(point, centroid, 2)
            distances.append(d)
        min_distance = min(distances)
        cluster_index = distances.index(min_distance)
        labels.append(cluster_index)
    return labels


def update_centroids(data, labels, k):
    new_centroids = []
    for cluster_num in range(k):
        cluster_points = []
        for i in range(len(data)):
            if labels[i] == cluster_num:
                cluster_points.append(data[i])

        if len(cluster_points) > 0:
            new_centroid = calculate_mean(cluster_points)
        else:
            new_centroid = random.choice(data)

        new_centroids.append(new_centroid)
    return new_centroids


def kmeans(data, k, max_iterations=50):
    random.seed(42)
    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        labels = assign_points(data, centroids)
        new_centroids = update_centroids(data, labels, k)

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return labels, centroids, iteration + 1


if __name__ == "__main__":
    data = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="marketing_campaign")
    data = data.dropna()

    numeric_columns = ["Income", "Recency", "MntWines", "MntFruits"]

    matrix = []
    for i in range(len(data)):
        row = []
        for col in numeric_columns:
            row.append(float(data.iloc[i][col]))
        matrix.append(row)
    num_cols = len(numeric_columns)
    col_min = []
    col_max = []
    for c in range(num_cols):
        column_values = [row[c] for row in matrix]
        col_min.append(min(column_values))
        col_max.append(max(column_values))

    normalized_matrix = []
    for row in matrix:
        new_row = []
        for c in range(num_cols):
            value = (row[c] - col_min[c]) / (col_max[c] - col_min[c])
            new_row.append(value)
        normalized_matrix.append(new_row)

    k = 3
    labels, centroids, iterations = kmeans(normalized_matrix, k)

    print("Converged in", iterations, "iterations")
    for cluster_num in range(k):
        count = labels.count(cluster_num)
        print("Cluster", cluster_num, "has", count, "points")