import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
df.replace("?", np.nan, inplace=True)

numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
binary_cols = ["on thyroxine", "query on thyroxine", "on antithyroid medication",
               "sick", "pregnant", "thyroid surgery", "I131 treatment",
               "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
               "tumor", "hypopituitary", "psych", "TSH measured", "T3 measured",
               "TT4 measured", "T4U measured", "FTI measured", "TBG measured"]
nominal_cols = ["sex", "referral source", "Condition"]

df_full = df.drop(columns=["Record ID"]).copy()
for col in numeric_cols:
    df_full[col] = pd.to_numeric(df_full[col], errors="coerce")
for col in binary_cols:
    df_full[col] = df_full[col].replace({"t": 1, "f": 0})
for col in nominal_cols:
    df_full[col] = df_full[col].astype("category").cat.codes

df_full = df_full.fillna(0)
N = 20
df20_full = df_full.iloc[:N]
df20_bin = df_full[binary_cols].iloc[:N]
def jaccard(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    denom = f01 + f10 + f11
    return f11 / denom if denom != 0 else 0.0

def smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    return (f11 + f00) / (f11 + f10 + f01 + f00)

def cosine_sim(v1, v2):
    num = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0.0

# similarity matrices
JC_mat = np.zeros((N, N))
SMC_mat = np.zeros((N, N))
COS_mat = np.zeros((N, N))

bin_vectors = df20_bin.to_numpy(dtype=float)
full_vectors = df20_full.to_numpy(dtype=float)

for i in range(N):
    for j in range(N):
        JC_mat[i, j] = jaccard(bin_vectors[i], bin_vectors[j])
        SMC_mat[i, j] = smc(bin_vectors[i], bin_vectors[j])
        COS_mat[i, j] = cosine_sim(full_vectors[i], full_vectors[j])

print("JC matrix (first 20 observations):\n", np.round(JC_mat, 2))
print("\nSMC matrix (first 20 observations):\n", np.round(SMC_mat, 2))
print("\nCOS matrix (first 20 observations):\n", np.round(COS_mat, 2))

# Heatmaps
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

sns.heatmap(JC_mat, annot=True, fmt=".2f", cmap="viridis", ax=axes[0], cbar=True)
axes[0].set_title("Jaccard Coefficient (JC) - first 20 obs (binary attrs)")

sns.heatmap(SMC_mat, annot=True, fmt=".2f", cmap="viridis", ax=axes[1], cbar=True)
axes[1].set_title("Simple Matching Coefficient (SMC) - first 20 obs (binary attrs)")

sns.heatmap(COS_mat, annot=True, fmt=".2f", cmap="viridis", ax=axes[2], cbar=True)
axes[2].set_title("Cosine Similarity (COS) - first 20 obs (all attrs)")

plt.tight_layout()
plt.savefig("A7_heatmaps.png", dpi=150)
print("\nHeatmaps saved as 'A7_heatmaps.png'")
plt.show()