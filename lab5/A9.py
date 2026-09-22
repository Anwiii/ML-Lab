import numpy as np
import matplotlib.pyplot as plt

from A8 import k_values, my_accuracy
from A8 import X_train, X_test, y_train, y_test


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

            # Calculate distance
            for i in range(len(self.X_train)):

                total = 0

                for j in range(len(test)):

                    total = total + (
                        self.X_train[i][j] - test[j]
                    ) ** 2

                d = total ** 0.5

                distances.append(
                    [d, self.y_train[i]]
                )

            # Sort
            distances.sort(
                key=lambda x: x[0]
            )

            # Take k neighbors
            neighbors = distances[:self.k]

            class_0 = 0
            class_1 = 0

            # Weighted voting
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

            # Select class
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


# -----------------------------------------
# Accuracy lists
# -----------------------------------------

weighted_accuracy = []
weighted_train_accuracy = []


# -----------------------------------------
# Test and training accuracy
# -----------------------------------------

for k in k_values:

    weighted_knn = WeightedKNN(k)

    weighted_knn.fit(
        X_train,
        y_train
    )

    # Test accuracy
    accuracy = weighted_knn.score(
        X_test,
        y_test
    )

    weighted_accuracy.append(accuracy)

    # Training accuracy
    train_accuracy = weighted_knn.score(
        X_train,
        y_train
    )

    weighted_train_accuracy.append(
        train_accuracy
    )


# -----------------------------------------
# Print results
# -----------------------------------------

print("k values:")
print(k_values)

print("Normal kNN accuracy:")
print(my_accuracy)

print("Weighted kNN accuracy:")
print(weighted_accuracy)

print("Weighted training accuracy:")
print(weighted_train_accuracy)


# -----------------------------------------
# Graph
# -----------------------------------------

plt.plot(
    k_values,
    my_accuracy,
    marker="o",
    label="Normal kNN"
)

plt.plot(
    k_values,
    weighted_accuracy,
    marker="o",
    label="Weighted kNN"
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("Normal kNN vs Weighted kNN")
plt.legend()
plt.show()