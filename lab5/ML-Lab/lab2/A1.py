import pandas as pd
import numpy as np
file_path = "Lab Session Data.xlsx"       
df = pd.read_excel(file_path, sheet_name="Purchase data")
df = df[["Customer", "Candies (#)", "Mangoes (Kg)","Milk Packets (#)", "Payment (Rs)"]].dropna()
#Segregate into X (features) and y (output)
X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].to_numpy(dtype=float)
y = df[["Payment (Rs)"]].to_numpy(dtype=float)
print("X (feature matrix):\n", X)
print("\ny (output vector):\n", y.ravel())
print("\nX shape:", X.shape)
print("y shape:", y.shape)

# Dimensionality of the vector space & number of vectors
dimensionality = X.shape[1]      # number of features -> dimension of each vector
num_vectors = X.shape[0]         # number of observations/rows
print("\nDimensionality of the vector space:", dimensionality)
print("Number of vectors in this vector space:", num_vectors)
rank_X = np.linalg.matrix_rank(X)
print("\nRank of matrix X:", rank_X)

# Cost of each product using Pseudo-Inverse (Xc = y  =>  c = pinv(X) @ y)
X_pinv = np.linalg.pinv(X)
c = X_pinv @ y
print("\nPseudo-Inverse of X:\n", X_pinv)
print("\nEstimated cost vector (Candy, Mango, Milk):\n", c.ravel())
print("Cost per unit:")
print(f"  Candy per piece      : Rs {c[0][0]:.2f}")
print(f"  Mango per Kg         : Rs {c[1][0]:.2f}")
print(f"  Milk packet per unit : Rs {c[2][0]:.2f}")