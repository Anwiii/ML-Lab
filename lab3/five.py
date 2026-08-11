import pandas as pd
import matplotlib.pyplot as plt
from four import minkowski_distance

data = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="marketing_campaign")
data = data.dropna()

numeric_columns = ["Year_Birth", "Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts"]
vector1 = []
vector2 = []
for col in numeric_columns:
    vector1.append(float(data.iloc[0][col]))
    vector2.append(float(data.iloc[1][col]))

p_values = []
distance_values = []

for p in range(1, 11):
    d = minkowski_distance(vector1, vector2, p)
    p_values.append(p)
    distance_values.append(d)
    print("p =", p, " distance =", d)

plt.plot(p_values, distance_values, marker="o")
plt.xlabel("p value")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance vs p")
plt.grid(True)
plt.savefig("A5_plot.png")
plt.show()