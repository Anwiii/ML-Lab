
import pandas as pd
import numpy as np
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
df.replace("?", np.nan, inplace=True)

numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

categorical_cols = ["sex", "referral source", "Condition"] + \
    ["on thyroxine", "query on thyroxine", "on antithyroid medication",
     "sick", "pregnant", "thyroid surgery", "I131 treatment",
     "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
     "tumor", "hypopituitary", "psych", "TSH measured", "T3 measured",
     "TT4 measured", "T4U measured", "FTI measured", "TBG measured"]
# Detect outliers (IQR method) to decide mean vs median per numeric col
def has_outliers_iqr(series, threshold_fraction=0.01):
    s = series.dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_outliers = ((s < lower) | (s > upper)).sum()
    return (n_outliers / len(s)) > threshold_fraction, n_outliers

print("=== Outlier check per numeric attribute ===")
impute_strategy = {}
for col in numeric_cols:
    has_out, n_out = has_outliers_iqr(df[col])
    strategy = "median" if has_out else "mean"
    impute_strategy[col] = strategy
    print(f"{col:6s} -> outliers: {n_out:5d}  -> strategy: {strategy}")
df_imputed = df.copy()

for col in numeric_cols:
    if impute_strategy[col] == "mean":
        fill_value = df_imputed[col].mean()
    else:
        fill_value = df_imputed[col].median()
    df_imputed[col] = df_imputed[col].fillna(fill_value)
    print(f"Filled '{col}' missing values with {impute_strategy[col]} = {fill_value:.4f}")
print("\n=== Categorical attribute imputation (mode) ===")
for col in categorical_cols:
    if df_imputed[col].isna().sum() > 0:
        mode_val = df_imputed[col].mode(dropna=True)[0]
        df_imputed[col] = df_imputed[col].fillna(mode_val)
        print(f"Filled '{col}' missing values with mode = '{mode_val}'")

print("\nMissing value count after imputation")
print(df_imputed.isna().sum()[df_imputed.isna().sum() > 0])
print("\n(If empty all missing values have been successfully imputed.)")

# Saving the imputed dataset for use in A9
df_imputed.to_csv("thyroid_imputed.csv", index=False)
print("\nImputed dataset saved as 'thyroid_imputed.csv'")