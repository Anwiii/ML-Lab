import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
file_path = "Lab Session Data.xlsx"          # update path if needed
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
df.replace("?", np.nan, inplace=True)

# ------------------------------------------------------------------
# 2. Identify binary (t/f) attributes and encode as 0/1
# ------------------------------------------------------------------
binary_cols = ["on thyroxine", "query on thyroxine", "on antithyroid medication",
               "sick", "pregnant", "thyroid surgery", "I131 treatment",
               "query hypothyroid", "query hyperthyroid", "lithium", "goitre",
               "tumor", "hypopituitary", "psych", "TSH measured", "T3 measured",
               "TT4 measured", "T4U measured", "FTI measured", "TBG measured"]

df_bin = df[binary_cols].replace({"t": 1, "f": 0})

# ------------------------------------------------------------------
# 3. Take the first 2 observation vectors (rows)
# ------------------------------------------------------------------
vec1 = df_bin.iloc[0].to_numpy(dtype=int)
vec2 = df_bin.iloc[1].to_numpy(dtype=int)

print("Binary attributes used:", binary_cols)
print("\nVector 1 (Record 0):", vec1)
print("Vector 2 (Record 1):", vec2)

# ------------------------------------------------------------------
# 4. Compute f11, f10, f01, f00
# ------------------------------------------------------------------
f11 = np.sum((vec1 == 1) & (vec2 == 1))
f10 = np.sum((vec1 == 1) & (vec2 == 0))
f01 = np.sum((vec1 == 0) & (vec2 == 1))
f00 = np.sum((vec1 == 0) & (vec2 == 0))

print(f"\nf11 (both 1) = {f11}")
print(f"f10 (1 in v1, 0 in v2) = {f10}")
print(f"f01 (0 in v1, 1 in v2) = {f01}")
print(f"f00 (both 0) = {f00}")

# ------------------------------------------------------------------
# 5. Jaccard Coefficient and Simple Matching Coefficient
# ------------------------------------------------------------------
JC = f11 / (f01 + f10 + f11)
SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

print(f"\nJaccard Coefficient (JC) = {JC:.4f}")
print(f"Simple Matching Coefficient (SMC) = {SMC:.4f}")

# ------------------------------------------------------------------
# 6. Observation / comparison
# ------------------------------------------------------------------
print("""
Observation:
- SMC counts BOTH matching 1s and matching 0s as agreement, so it is
  strongly influenced by f00 (attributes that are simply absent/false
  in both records). In this dataset most binary flags (sick, pregnant,
  goitre, lithium, etc.) are 'f' for most patients, so f00 dominates
  and SMC comes out high even if the two patients don't truly share
  many meaningful (positive) symptoms/conditions.

- JC ignores f00 entirely and only measures similarity based on
  attributes that are actually present (1) in at least one of the two
  vectors. This makes JC more meaningful here, since a shared absence
  of a rare condition (both patients NOT pregnant, NOT having goitre,
  etc.) is not informative similarity - it's just the common case.

- Hence for sparse/asymmetric binary attributes like these (where 0 is
  the overwhelmingly common value and doesn't indicate real similarity),
  JC is the more appropriate measure. SMC would be preferred only if
  both 0 and 1 carried equally meaningful information (symmetric binary
  attributes), which is not the case here.
""")