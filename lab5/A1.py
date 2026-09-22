import pandas as pd
import numpy as np
data = pd.read_csv("merged_dataset.csv")


def encode_data(data):

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

        for j in range(0, n-i-1):

            if data[j][0] > data[j+1][0]:

                temp = data[j]
                data[j] = data[j+1]
                data[j+1] = temp

    return data


def selection_sort(data):

    n = len(data)

    for i in range(n):

        small = i

        for j in range(i+1, n):

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

            data[j+1] = data[j]
            j = j - 1

        data[j+1] = temp

    return data


def find_neighbors(X_train, y_train, test, k,
                   distance_type="euclidean",
                   sort_type="bubble"):

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


    # Majority voting
    if class_0 > class_1:
        return 0

    elif class_1 > class_0:
        return 1

    else:
        # Tie breaking
        # Choose the class of the closest neighbor
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

data = data.drop(columns=["participant_id", "date"])

data = encode_data(data)

data = fill_missing(data, "mean")


X = data.drop(columns=["high_fatigue"])
y = data["high_fatigue"]

print("Number of rows:", len(X))
print("Number of features:", len(X.columns))




