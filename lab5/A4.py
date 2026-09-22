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
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("training done")

#a5
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy * 100, "%")
#a6
prediction = neigh.predict(X_test)
print("Predicted classes:")
print(prediction)