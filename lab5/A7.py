import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv("merged_dataset.csv")
data = data.drop(columns=["participant_id", "date"])
for column in data.columns:
    if data[column].isnull().sum() > 0:
        data[column] = data[column].fillna(data[column].mean())
X = data.drop(columns=["high_fatigue"])
y = data["high_fatigue"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
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
                    total = total + (
                        self.X_train[i][j] - test[j]
                    ) ** 2
                d = total ** 0.5
                distances.append(
                    [d, self.y_train[i]]
                )
            distances.sort(
                key=lambda x: x[0]
            )
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
my_knn = MyKNN(k=3)
my_knn.fit(X_train, y_train)

prediction = my_knn.predict(X_test)

accuracy = my_knn.score(X_test, y_test)


print("Predictions:")
print(prediction)

print("Accuracy:")
print(accuracy)