import pandas as pd
import numpy as np
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
df.replace("?", np.nan, inplace=True)

# ------------------------------------------------------------------
# 2. Build a fully numeric version of the dataframe
#    - numeric columns -> as-is
#    - binary t/f columns -> 1/0
#    - other categorical columns (sex, referral source, Condition) -> label encoded
#    - Record ID dropped (identifier, not a feature)
# ------------------------------------------------------------------
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
# For this similarity computation, fill any missing values with 0
# (placeholder only -> proper imputation is done later in A8)
df_full = df_full.fillna(0)
A = df_full.iloc[0].to_numpy(dtype=float)
B = df_full.iloc[1].to_numpy(dtype=float)
print("Vector A (Record 0), first 10 values:", A[:10])
print("Vector B (Record 1), first 10 values:", B[:10])
print("Vector length (num attributes):", len(A))
# Cosine similarity: cos(A,B) = <A,B> / (||A|| * ||B||)-
dot_product = np.dot(A, B)
norm_A = np.linalg.norm(A)
norm_B = np.linalg.norm(B)
cos_sim = dot_product / (norm_A * norm_B)
print(f"\nDot product <A,B>  = {dot_product:.4f}")
print(f"||A||               = {norm_A:.4f}")
print(f"||B||               = {norm_B:.4f}")
print(f"Cosine Similarity    = {cos_sim:.4f}")