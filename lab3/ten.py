# A10 - Histogram of a feature, with mean and variance

import pandas as pd
import matplotlib.pyplot as plt
from eight import calculate_mean, calculate_variance

data = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="marketing_campaign")
data = data.dropna()

income_values = list(data["Income"])

plt.hist(income_values, bins=10, edgecolor="black")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Histogram of Income")
plt.savefig("A10_histogram.png")
plt.show()

# our mean/variance functions expect a matrix (list of rows),
# so wrap each value in its own small row
income_matrix = []
for value in income_values:
    income_matrix.append([value])

mean_result = calculate_mean(income_matrix)
variance_result = calculate_variance(income_matrix)

print("Mean Income    :", mean_result[0])
print("Variance Income:", variance_result[0])