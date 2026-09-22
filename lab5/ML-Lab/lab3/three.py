# A3 - Apply encoding to the marketing_campaign dataset and check dimensionality

import pandas as pd
from two import label_encode, one_hot_encode

data = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="marketing_campaign")
data = data.dropna()

print("Shape before encoding (rows, columns):", data.shape)

# Education has a natural order -> label encode
education_column = list(data["Education"])
education_encoded, education_categories = label_encode(education_column)

# Marital_Status has no order -> one-hot encode
marital_column = list(data["Marital_Status"])
marital_encoded, marital_categories = one_hot_encode(marital_column)

print("\nEducation categories found:", education_categories)
print("Marital status categories found:", marital_categories)

# columns before, minus ID and Dt_Customer, minus the original Marital_Status
# column, plus one new column for every marital category
columns_before = len(data.columns)
columns_after = columns_before - 2 - 1 + len(marital_categories)

print("\nNumber of columns before encoding:", columns_before)
print("Number of columns after encoding :", columns_after)