import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_excel("Lab Session Data (1).xlsx", sheet_name="marketing_campaign")

cols_to_ignore = ['ID', 'Dt_Customer'] 
df_clean = df.drop(columns=[col for col in cols_to_ignore if col in df.columns])

categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna('Unknown').astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' 
)
encoded_array = preprocessor.fit_transform(df_clean)

encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
numerical_cols = [col for col in df_clean.columns if col not in categorical_cols]
all_column_names = list(encoded_feature_names) + numerical_cols
final_df = pd.DataFrame(encoded_array, columns=all_column_names)

print( final_df.shape)
print(final_df.head())
