"""
knn_pipeline.py
================
Modular k-NN classifier -- all-in-one file.

Sections below (still fully modular internally -- each is an independent
class/function group, just kept in one file for convenience):

  0. Config              -> KNNConfig
  a. Encoding             -> Encoder
  b. Data Imputation      -> Imputer
  c. Distance Calculation -> DistanceCalculator
  d. Sorting              -> sort_by_distance (bubble / merge / quick)
  e. Identify Neighbors   -> get_k_nearest (with tie-breaking)
  f. Class Evaluation     -> majority_vote (with tie-breaking)
  +  Orchestrator         -> KNNClassifier (fit/predict/score)
  +  Utils                -> train_test_split_df, confusion_counts
  +  Demo                 -> loads "Lab Session Data (1).xlsx" /
                              sheet "marketing_campaign" and runs it all

Run directly:
    python knn_pipeline.py
"""

import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =====================================================================
# 0. CONFIG -- single place that drives every stage of the pipeline
# =====================================================================
@dataclass
class KNNConfig:
    # ---- target / features ----
    target_column: str = None
    feature_columns: Optional[List[str]] = None   # None => use all other columns

    # ---- (a) Encoding ----
    encoding_strategy: str = "label"               # "label" | "onehot" | "none"

    # ---- (b) Imputation ----
    numeric_impute_strategy: str = "mean"           # "mean" | "median" | "mode"
    categorical_impute_strategy: str = "mode"

    # ---- (c) Distance ----
    distance_metric: str = "euclidean"              # "euclidean"|"manhattan"|"minkowski"|"chebyshev"|"hamming"
    minkowski_p: float = 3

    # ---- (d) Sorting ----
    sorting_algorithm: str = "quick"                # "bubble" | "merge" | "quick"

    # ---- (e) Neighbor identification ----
    k: int = 5
    neighbor_tie_breaking: str = "include_all"      # "include_all" | "lowest_index"

    # ---- (f) Voting ----
    voting_tie_breaking: str = "nearest"            # "nearest" | "lowest_label"

    # ---- misc ----
    random_seed: int = 42


# =====================================================================
# (a) ENCODING -- categorical -> numeric
# =====================================================================
def get_categorical_columns(df: pd.DataFrame):
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


class LabelEncoderCustom:
    """category -> integer code. fit on train, reused on test (unseen -> -1)."""

    def __init__(self):
        self.mapping_ = {}
        self.inverse_mapping_ = {}

    def fit(self, series: pd.Series):
        categories = sorted(series.dropna().unique().tolist(), key=str)
        self.mapping_ = {cat: idx for idx, cat in enumerate(categories)}
        self.inverse_mapping_ = {idx: cat for cat, idx in self.mapping_.items()}
        return self

    def transform(self, series: pd.Series):
        return series.map(self.mapping_).fillna(-1).astype(float)

    def fit_transform(self, series: pd.Series):
        return self.fit(series).transform(series)


class OneHotEncoderCustom:
    """One-hot via pandas.get_dummies; fit stores column set so train/test align."""

    def __init__(self):
        self.columns_ = None

    def fit_transform(self, df: pd.DataFrame, cat_cols):
        dummies = pd.get_dummies(df[cat_cols], columns=cat_cols, dtype=float)
        self.columns_ = dummies.columns.tolist()
        rest = df.drop(columns=cat_cols)
        return pd.concat([rest, dummies], axis=1)

    def transform(self, df: pd.DataFrame, cat_cols):
        dummies = pd.get_dummies(df[cat_cols], columns=cat_cols, dtype=float)
        dummies = dummies.reindex(columns=self.columns_, fill_value=0.0)
        rest = df.drop(columns=cat_cols)
        return pd.concat([rest, dummies], axis=1)


class Encoder:
    """Facade. strategy: 'label' | 'onehot' | 'none'."""

    def __init__(self, strategy: str = "label"):
        self.strategy = strategy
        self._label_encoders = {}
        self._onehot = None
        self._cat_cols = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self._cat_cols = get_categorical_columns(df)

        if self.strategy == "none" or not self._cat_cols:
            return df
        if self.strategy == "label":
            for col in self._cat_cols:
                enc = LabelEncoderCustom()
                df[col] = enc.fit_transform(df[col])
                self._label_encoders[col] = enc
            return df
        if self.strategy == "onehot":
            self._onehot = OneHotEncoderCustom()
            return self._onehot.fit_transform(df, self._cat_cols)
        raise ValueError(f"Unknown encoding strategy: {self.strategy}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.strategy == "none" or not self._cat_cols:
            return df
        if self.strategy == "label":
            for col in self._cat_cols:
                df[col] = self._label_encoders[col].transform(df[col])
            return df
        if self.strategy == "onehot":
            return self._onehot.transform(df, self._cat_cols)
        raise ValueError(f"Unknown encoding strategy: {self.strategy}")


# =====================================================================
# (b) DATA IMPUTATION -- fill missing values with central tendency
# =====================================================================
class Imputer:
    def __init__(self, numeric_strategy: str = "mean", categorical_strategy: str = "mode"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_values_ = {}

    @staticmethod
    def _mode(series: pd.Series):
        m = series.mode(dropna=True)
        return m.iloc[0] if not m.empty else np.nan

    def _compute_fill(self, series: pd.Series, is_numeric: bool):
        strategy = self.numeric_strategy if is_numeric else self.categorical_strategy
        if strategy == "mean":
            return series.mean()
        if strategy == "median":
            return series.median()
        if strategy == "mode":
            return self._mode(series)
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    def fit(self, df: pd.DataFrame):
        self.fill_values_ = {}
        for col in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            self.fill_values_[col] = self._compute_fill(df[col], is_numeric)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].isna().any():
                fill = self.fill_values_.get(col, None)
                if fill is not None and not (isinstance(fill, float) and np.isnan(fill)):
                    df[col] = df[col].fillna(fill)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# =====================================================================
# (c) DISTANCE CALCULATION
# =====================================================================
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def minkowski(a: np.ndarray, b: np.ndarray, p: float = 3) -> float:
    return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))


def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


class DistanceCalculator:
    def __init__(self, metric: str = "euclidean", p: float = 3):
        self.metric = metric
        self.p = p

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "minkowski":
            return minkowski(a, b, self.p)
        fn = {"euclidean": euclidean, "manhattan": manhattan,
              "chebyshev": chebyshev, "hamming": hamming}.get(self.metric)
        if fn is None:
            raise ValueError(f"Unknown distance metric: {self.metric}")
        return fn(a, b)

    def compute_all(self, query: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Vectorised distance from one query point to every row in `references`."""
        if self.metric == "euclidean":
            return np.sqrt(np.sum((references - query) ** 2, axis=1))
        if self.metric == "manhattan":
            return np.sum(np.abs(references - query), axis=1)
        if self.metric == "minkowski":
            return np.sum(np.abs(references - query) ** self.p, axis=1) ** (1.0 / self.p)
        if self.metric == "chebyshev":
            return np.max(np.abs(references - query), axis=1)
        if self.metric == "hamming":
            return np.mean(references != query, axis=1)
        raise ValueError(f"Unknown distance metric: {self.metric}")


# =====================================================================
# (d) SORTING -- 3 DSA algorithms, selectable via config
# =====================================================================
Pair = Tuple[float, int]   # (distance, original_index)


def bubble_sort(pairs: List[Pair]) -> List[Pair]:
    """O(n^2), stable."""
    arr = pairs.copy()
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def merge_sort(pairs: List[Pair]) -> List[Pair]:
    """O(n log n), stable."""
    if len(pairs) <= 1:
        return pairs.copy()
    mid = len(pairs) // 2
    left = merge_sort(pairs[:mid])
    right = merge_sort(pairs[mid:])
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i][0] <= right[j][0]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(pairs: List[Pair]) -> List[Pair]:
    """O(n log n) average. Partition-by-value keeps it stable for ties."""
    if len(pairs) <= 1:
        return pairs.copy()
    pivot = pairs[len(pairs) // 2][0]
    less = [p for p in pairs if p[0] < pivot]
    equal = [p for p in pairs if p[0] == pivot]
    greater = [p for p in pairs if p[0] > pivot]
    return quick_sort(less) + equal + quick_sort(greater)


_SORT_ALGORITHMS = {"bubble": bubble_sort, "merge": merge_sort, "quick": quick_sort}


def sort_by_distance(pairs: List[Pair], algorithm: str = "quick") -> List[Pair]:
    if algorithm not in _SORT_ALGORITHMS:
        raise ValueError(f"Unknown sorting algorithm '{algorithm}'. Choose from {list(_SORT_ALGORITHMS)}")
    return _SORT_ALGORITHMS[algorithm](pairs)


# =====================================================================
# (e) IDENTIFY NEIGHBORS -- k nearest, with tie-breaking
# =====================================================================
def get_k_nearest(sorted_pairs: List[Pair], k: int, tie_breaking: str = "include_all") -> List[Pair]:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k >= len(sorted_pairs):
        return sorted_pairs.copy()

    boundary_distance = sorted_pairs[k - 1][0]

    if tie_breaking == "lowest_index":
        head = sorted_pairs[:k]
        tail_ties = [p for p in sorted_pairs[k:] if p[0] == boundary_distance]
        if tail_ties:
            boundary_group = [p for p in head if p[0] == boundary_distance] + tail_ties
            boundary_group_sorted = sorted(boundary_group, key=lambda p: p[1])
            n_slots = len([p for p in head if p[0] == boundary_distance])
            kept = boundary_group_sorted[:n_slots]
            head = [p for p in head if p[0] != boundary_distance] + kept
            head = sorted(head, key=lambda p: p[0])
        return head

    if tie_breaking == "include_all":
        neighbors = list(sorted_pairs[:k])
        idx = k
        while idx < len(sorted_pairs) and sorted_pairs[idx][0] == boundary_distance:
            neighbors.append(sorted_pairs[idx])
            idx += 1
        return neighbors

    raise ValueError(f"Unknown neighbor tie-breaking strategy: {tie_breaking}")


# =====================================================================
# (f) CLASS EVALUATION AND ASSIGNMENT -- majority vote, with tie-breaking
# =====================================================================
def majority_vote(neighbor_pairs: List[Pair], y_train: Sequence, tie_breaking: str = "nearest"):
    labels = [y_train[idx] for (_dist, idx) in neighbor_pairs]
    counts = Counter(labels)
    max_votes = max(counts.values())
    tied_classes = [cls for cls, cnt in counts.items() if cnt == max_votes]

    if len(tied_classes) == 1:
        return tied_classes[0]

    if tie_breaking == "nearest":
        best_class, best_dist = None, float("inf")
        for dist, idx in neighbor_pairs:
            label = y_train[idx]
            if label in tied_classes and dist < best_dist:
                best_dist = dist
                best_class = label
        return best_class

    if tie_breaking == "lowest_label":
        return sorted(tied_classes, key=lambda c: str(c))[0]

    raise ValueError(f"Unknown voting tie-breaking strategy: {tie_breaking}")


# =====================================================================
# ORCHESTRATOR -- ties (a) through (f) together, sklearn-style API
# =====================================================================
class KNNClassifier:
    def __init__(self, config: KNNConfig):
        self.config = config
        self.encoder = Encoder(strategy=config.encoding_strategy)
        self.imputer = Imputer(
            numeric_strategy=config.numeric_impute_strategy,
            categorical_strategy=config.categorical_impute_strategy,
        )
        self.distance_calc = DistanceCalculator(metric=config.distance_metric, p=config.minkowski_p)
        self._X_train = None
        self._y_train = None
        self._feature_names = None

    def _preprocess_fit(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer.fit_transform(X)
        X_encoded = self.encoder.fit_transform(X_imputed)
        self._feature_names = X_encoded.columns.tolist()
        return X_encoded.to_numpy(dtype=float)

    def _preprocess_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_imputed = self.imputer.transform(X)
        X_encoded = self.encoder.transform(X_imputed)
        X_encoded = X_encoded.reindex(columns=self._feature_names, fill_value=0.0)
        return X_encoded.to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._X_train = self._preprocess_fit(X)
        self._y_train = list(y.reset_index(drop=True))
        return self

    def _predict_one(self, query: np.ndarray):
        distances = self.distance_calc.compute_all(query, self._X_train)
        pairs = [(d, i) for i, d in enumerate(distances)]     # (distance, index)
        sorted_pairs = sort_by_distance(pairs, algorithm=self.config.sorting_algorithm)
        neighbor_pairs = get_k_nearest(sorted_pairs, k=self.config.k,
                                        tie_breaking=self.config.neighbor_tie_breaking)
        predicted_label = majority_vote(neighbor_pairs, self._y_train,
                                         tie_breaking=self.config.voting_tie_breaking)
        return predicted_label, neighbor_pairs

    def predict(self, X: pd.DataFrame):
        X_processed = self._preprocess_transform(X)
        return np.array([self._predict_one(row)[0] for row in X_processed])

    def predict_with_neighbors(self, X: pd.DataFrame):
        X_processed = self._preprocess_transform(X)
        preds, neighbor_info = [], []
        for row in X_processed:
            label, neighbors = self._predict_one(row)
            preds.append(label)
            neighbor_info.append(neighbors)
        return np.array(preds), neighbor_info

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        preds = self.predict(X)
        y_true = np.array(list(y))
        return float(np.mean(preds == y_true))


# =====================================================================
# UTILS -- reused from earlier lab sessions
# =====================================================================
def train_test_split_df(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    n = len(X)
    indices = rng.permutation(n)
    n_test = int(n * test_size)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    return X_train, X_test, y_train, y_test


def confusion_counts(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / len(y_true) if y_true else 0.0
    return {"accuracy": accuracy, "n": len(y_true), "correct": correct}


# =====================================================================
# DEMO -- loads the lab Excel file and runs the full pipeline
# =====================================================================
EXCEL_PATH = "Lab Session Data (1).xlsx"
SHEET_NAME = "marketing_campaign"
TARGET_COLUMN_CANDIDATES = ["Response", "response", "Target", "target", "Class", "class"]


def _pick_target_column(df: pd.DataFrame) -> str:
    for cand in TARGET_COLUMN_CANDIDATES:
        if cand in df.columns:
            return cand
    return df.columns[-1]


def _build_id_drop_list(df: pd.DataFrame):
    drop_candidates = ["ID", "Id", "id", "Customer_ID", "Dt_Customer"]
    return [c for c in drop_candidates if c in df.columns]


def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"[!] Could not find '{EXCEL_PATH}' in the current directory.")
        print("    Place the file next to this script (or edit EXCEL_PATH) and re-run.")
        return

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    print(f"Loaded sheet '{SHEET_NAME}' with shape {df.shape}")

    target_col = _pick_target_column(df)
    print(f"Using target column: '{target_col}'")

    drop_cols = _build_id_drop_list(df)
    if drop_cols:
        print(f"Dropping identifier columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split_df(X, y, test_size=0.2, random_seed=42)

    # Show the config-driven modularity: swap sorting algo / distance metric / tie-break rule
    configs_to_try = [
        ("quick",  "euclidean", "nearest"),
        ("merge",  "manhattan", "nearest"),
        ("bubble", "euclidean", "lowest_label"),
    ]

    for sort_algo, metric, vote_tie in configs_to_try:
        cfg = KNNConfig(
            target_column=target_col,
            encoding_strategy="label",
            numeric_impute_strategy="mean",
            categorical_impute_strategy="mode",
            distance_metric=metric,
            sorting_algorithm=sort_algo,
            k=5,
            neighbor_tie_breaking="include_all",
            voting_tie_breaking=vote_tie,
        )
        model = KNNClassifier(cfg)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        result = confusion_counts(y_test, preds)
        print(
            f"[sort={sort_algo:6s} | dist={metric:9s} | vote_tie={vote_tie:12s}] "
            f"accuracy = {result['accuracy']:.4f}  ({result['correct']}/{result['n']})"
        )


if __name__ == "__main__":
    main()
