import pandas as pd

data = pd.read_csv("wellness.csv")
print(data["effective_time_frame"].head(10))