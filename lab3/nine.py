# A9 - Compare our own mean/std functions with numpy's built-in ones

import pandas as pd
import numpy as np
from eight import calculate_mean, calculate_std

data = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="marketing_campaign")
data = data.dropna()

numeric_columns = ["Year_Birth", "Income", "Recency", "MntWines"]

matrix = []
for i in range(len(data)):
    row = []
    for col in numeric_columns:
        row.append(float(data.iloc[i][col]))
    matrix.append(row)

my_mean = calculate_mean(matrix)
my_std = calculate_std(matrix)

numpy_array = np.array(matrix)
numpy_mean = numpy_array.mean(axis=0)
numpy_std = numpy_array.std(axis=0)

print("Columns used:", numeric_columns)
print()
print("My mean   :", my_mean)
print("Numpy mean:", list(numpy_mean))
print()
print("My std    :", my_std)
print("Numpy std :", list(numpy_std))