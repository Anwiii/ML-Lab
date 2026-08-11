# encodings.py
# A2 & A3 — Label Encoding and One-Hot Encoding

import numpy as np
import pandas as pd


def label_encode(column):
    """
    Encode a categorical Series as integer labels (0, 1, 2, ...).
    Categories are sorted alphabetically for a deterministic mapping.
    Returns the encoded Series and the mapping dictionary.
    """
    unique_vals = sorted(column.dropna().unique())
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = column.map(mapping)
    return encoded, mapping


def one_hot_encode(df, column):
    """
    One-hot encode a single categorical column.
    Drops the original column and appends binary indicator columns.
    Returns the modified dataframe.
    """
    unique_vals = sorted(df[column].dropna().unique())
    for val in unique_vals:
        df[f"{column}_{val}"] = (df[column] == val).astype(int)
    df = df.drop(columns=[column])
    return df


def apply_encoding(df):
    """
    Apply encoding to the marketing_campaign dataset.
    - Label encodes Education (ordinal).
    - One-hot encodes Marital_Status (nominal).
    - Drops non-numeric/non-useful columns.
    Returns the encoded dataframe and the education mapping.
    """
    df_enc = df.copy()

    # Drop columns not useful for numeric analysis
    df_enc = df_enc.drop(columns=["ID", "Dt_Customer"], errors="ignore")
    df_enc = df_enc.dropna()

    # Label encode Education (has ordinal meaning)
    df_enc["Education"], edu_map = label_encode(df_enc["Education"])

    # One-hot encode Marital_Status (purely nominal)
    df_enc = one_hot_encode(df_enc, "Marital_Status")

    return df_enc, edu_map
