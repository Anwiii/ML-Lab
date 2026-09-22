import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("merged_dataset.csv")
data = data.drop(columns=["participant_id", "date"])
for column in data.columns:
    if data[column].isnull().sum() > 0:
        data[column] = data[column].fillna(data[column].mean())
X = data.drop(columns=["high_fatigue"])
y = data["high_fatigue"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
print("Training :", len(X_train))
print("Testing :", len(X_test))