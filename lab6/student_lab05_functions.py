"""
student_lab05_functions.py
===========================
Anwi's own Lab 5 functions and classes (from A1.py / A7.py / A9.py),
extracted here WITHOUT the top-level script code (the csv read + prints
that ran at import time in the original files), so they can be imported
cleanly by test_lab06.py and performance_comparison.py.

Nothing in the logic below has been changed from the original submission.
"""

import numpy as np


# ---- from A1.py ----
def encode_data(data):
    import pandas as pd
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = pd.factorize(data[column])[0]
    return data


def fill_missing(data, method="mean"):
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if method == "mean":
                data[column] = data[column].fillna(data[column].mean())
            elif method == "median":
                data[column] = data[column].fillna(data[column].median())
            elif method == "mode":
                data[column] = data[column].fillna(data[column].mode()[0])
    return data


def distance(x1, x2, method="euclidean"):
    if method == "euclidean":
        total = 0
        for i in range(len(x1)):
            total = total + (x1[i] - x2[i]) ** 2
        return total ** 0.5
    elif method == "manhattan":
        total = 0
        for i in range(len(x1)):
            total = total + abs(x1[i] - x2[i])
        return total


def bubble_sort(data):
    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j][0] > data[j + 1][0]:
                temp = data[j]
                data[j] = data[j + 1]
                data[j + 1] = temp
    return data


def selection_sort(data):
    n = len(data)
    for i in range(n):
        small = i
        for j in range(i + 1, n):
            if data[j][0] < data[small][0]:
                small = j
        temp = data[i]
        data[i] = data[small]
        data[small] = temp
    return data


def insertion_sort(data):
    for i in range(1, len(data)):
        temp = data[i]
        j = i - 1
        while j >= 0 and data[j][0] > temp[0]:
            data[j + 1] = data[j]
            j = j - 1
        data[j + 1] = temp
    return data


def find_neighbors(X_train, y_train, test, k,
                    distance_type="euclidean", sort_type="bubble"):
    distances = []
    for i in range(len(X_train)):
        d = distance(X_train[i], test, distance_type)
        distances.append([d, y_train[i], i])

    if sort_type == "bubble":
        distances = bubble_sort(distances)
    elif sort_type == "selection":
        distances = selection_sort(distances)
    elif sort_type == "insertion":
        distances = insertion_sort(distances)

    neighbors = distances[:k]
    return neighbors


def find_class(neighbors):
    class_0 = 0
    class_1 = 0
    for neighbor in neighbors:
        if neighbor[1] == 0:
            class_0 = class_0 + 1
        else:
            class_1 = class_1 + 1

    if class_0 > class_1:
        return 0
    elif class_1 > class_0:
        return 1
    else:
        return neighbors[0][1]


def weighted_class(neighbors):
    class_0 = 0
    class_1 = 0
    for neighbor in neighbors:
        d = neighbor[0]
        label = neighbor[1]
        if d == 0:
            weight = 100000
        else:
            weight = 1 / d
        if label == 0:
            class_0 = class_0 + weight
        else:
            class_1 = class_1 + weight

    if class_0 > class_1:
        return 0
    elif class_1 > class_0:
        return 1
    else:
        return neighbors[0][1]


# ---- from A7.py ----
class MyKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for test in X:
            distances = []
            for i in range(len(self.X_train)):
                total = 0
                for j in range(len(test)):
                    total = total + (self.X_train[i][j] - test[j]) ** 2
                d = total ** 0.5
                distances.append([d, self.y_train[i]])
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            class_0 = 0
            class_1 = 0
            for neighbor in neighbors:
                if neighbor[1] == 0:
                    class_0 = class_0 + 1
                else:
                    class_1 = class_1 + 1
            if class_0 > class_1:
                result = 0
            elif class_1 > class_0:
                result = 1
            else:
                result = neighbors[0][1]
            predictions.append(result)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        correct = 0
        for i in range(len(y)):
            if predictions[i] == y.iloc[i]:
                correct = correct + 1
        accuracy = correct / len(y)
        return accuracy


# ---- from A9.py ----
class WeightedKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for test in X:
            distances = []
            for i in range(len(self.X_train)):
                total = 0
                for j in range(len(test)):
                    total = total + (self.X_train[i][j] - test[j]) ** 2
                d = total ** 0.5
                distances.append([d, self.y_train[i]])
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            class_0 = 0
            class_1 = 0
            for neighbor in neighbors:
                d = neighbor[0]
                label = neighbor[1]
                if d == 0:
                    weight = 100000
                else:
                    weight = 1 / d
                if label == 0:
                    class_0 = class_0 + weight
                else:
                    class_1 = class_1 + weight
            if class_0 > class_1:
                result = 0
            elif class_1 > class_0:
                result = 1
            else:
                result = neighbors[0][1]
            predictions.append(result)
        return np.array(predictions)

    def score(self, X, y):
        predictions = self.predict(X)
        correct = 0
        for i in range(len(y)):
            if predictions[i] == y.iloc[i]:
                correct = correct + 1
        accuracy = correct / len(y)
        return accuracy
