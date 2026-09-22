
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
file_path = "Lab Session Data.xlsx"         
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
# The dataset uses '?' as the missing-value marker -> replace with NaN
df.replace("?", np.nan, inplace=True)
# Identify numeric vs categorical attributes
# Attributes that are numeric but stored as text because of '?' markers
numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
binary_cols = ["on thyroxine", "query on thyroxine", "on antithyroid medication",
               "sick", "pregnant", "thyroid surgery", "I131 treatment",
               "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
               "tumor", "hypopituitary", "psych", "TSH measured", "T3 measured",
               "TT4 measured", "T4U measured", "FTI measured", "TBG measured"]

# Other categorical (nominal) attributes
nominal_cols = ["sex", "referral source", "Condition"]
id_cols = ["Record ID"]  
print("=" * 70)
print("1. ATTRIBUTE TYPE SUMMARY")
print("=" * 70)
print(f"\nIdentifier attribute       : {id_cols}")
print(f"\nNumeric (interval/ratio)   : {numeric_cols}")
print(f"\nBinary/nominal (t/f flags) : {binary_cols}")
print(f"\nOther nominal categorical  : {nominal_cols}")
print("\nNote: There is no natural ORDER among categories for sex, referral")
print("source, Condition, or the t/f flags -> all are NOMINAL, not ordinal.")
print("age is a ratio-scale numeric variable; the lab test values")
print("(TSH, T3, TT4, T4U, FTI, TBG) are also ratio-scale numeric variables.")
# Data range for numeric variables
print("=" * 70)
print("3. DATA RANGE FOR NUMERIC VARIABLES")
print("=" * 70)
range_summary = pd.DataFrame({
    "min": df[numeric_cols].min(),
    "max": df[numeric_cols].max(),
    "range": df[numeric_cols].max() - df[numeric_cols].min()
})
print(range_summary)

# Missing values in each attribute
print("\n" + "=" * 70)
print("4. MISSING VALUES PER ATTRIBUTE")
print("=" * 70)
missing_counts = df.isna().sum()
missing_pct = (df.isna().sum() / len(df) * 100).round(2)
missing_summary = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_%": missing_pct
})
missing_summary = missing_summary[missing_summary["missing_count"] > 0]
missing_summary = missing_summary.sort_values("missing_count", ascending=False)
print(missing_summary)

# Outlier detection (IQR method) for numeric variables
print("\n" + "=" * 70)
print("5. OUTLIER DETECTION (IQR method)")
print("=" * 70)
def count_outliers_iqr(series):
    s = series.dropna()
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = s[(s < lower) | (s > upper)]
    return len(outliers), lower, upper
outlier_summary = []
for col in numeric_cols:
    n_out, lower, upper = count_outliers_iqr(df[col])
    outlier_summary.append({
        "attribute": col,
        "num_outliers": n_out,
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2)
    })
outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))

# Mean and variance/std for numeric variables
print("\n" + "=" * 70)
print("6. MEAN AND VARIANCE (STD) FOR NUMERIC VARIABLES")
print("=" * 70)

stats_summary = pd.DataFrame({
    "mean": df[numeric_cols].mean(),
    "variance": df[numeric_cols].var(),
    "std_dev": df[numeric_cols].std()
})
print(stats_summary.round(4))