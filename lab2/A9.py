import pandas as pd
import numpy as np
df = pd.read_csv("thyroid_imputed.csv")
numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
print("=== Range of numeric attributes (before normalization) ===")
range_summary = pd.DataFrame({
    "min": df[numeric_cols].min(),
    "max": df[numeric_cols].max(),
    "range": df[numeric_cols].max() - df[numeric_cols].min()
})
print(range_summary)
#minmax normalization
df_minmax = df.copy()
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    df_minmax[col] = (df[col] - col_min) / (col_max - col_min)

print("=== Min-Max normalized numeric attributes (first 5 rows) ===")
print(df_minmax[numeric_cols].head())
#z-score normalization 
df_zscore = df.copy()
for col in numeric_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    df_zscore[col] = (df[col] - mean_val) / std_val
print("\n=== Z-score normalized numeric attributes (first 5 rows) ===")
print(df_zscore[numeric_cols].head())
df_minmax.to_csv("thyroid_normalized_minmax.csv", index=False)
df_zscore.to_csv("thyroid_normalized_zscore.csv", index=False)
print("\nSaved 'thyroid_normalized_minmax.csv' and 'thyroid_normalized_zscore.csv'")
