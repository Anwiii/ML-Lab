"""
ai_knn.py
=========
Lab 06 (23CSE301) - AI-generated k-Nearest Neighbors implementation.

GenAI Tool used for this entire file: Claude (Anthropic), Sonnet model,
via claude.ai chat interface, dated 2026-09-22.

Design notes (why this is NOT a copy of the Lab 5 submission):
  - Distances are computed with vectorized NumPy broadcasting (one call per
    query point covers the whole training set) instead of nested Python
    `for` loops over rows and features.
  - Three distance metrics are implemented through a single generic
    Minkowski-distance function (p=1 -> Manhattan, p=2 -> Euclidean),
    plus Chebyshev, instead of separate hand-written accumulator loops.
  - Sorting is offered as three *from-scratch* algorithms that were not
    used in Lab 5 (merge sort, quicksort, heap sort via `heapq`) operating
    on index/key pairs, selectable through a config string, alongside a
    NumPy `argsort` fast path.
  - Tie-breaking at the k-th neighbor boundary is handled by first
    including *every* training point exactly tied with the k-th nearest
    distance (so no point that is genuinely as close as the last accepted
    neighbor is silently dropped), and a class-count tie is then broken by
    falling back to an inverse-distance-weighted vote among only the tied
    classes (rather than simply taking the single nearest neighbor's
    label).
  - Uniform-vote kNN and distance-weighted kNN are NOT two separate
    classes; they are a single `AIKNNClassifier` with a `weights=`
    constructor argument, mirroring scikit-learn's own API
    (`KNeighborsClassifier(weights="uniform"|"distance")`).
  - Preprocessing (encoding / imputation) reuses well-tested library
    primitives (`sklearn.preprocessing.LabelEncoder`,
    `sklearn.impute.SimpleImputer`) rather than hand-rolled loops.
"""

from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# ----------------------------------------------------------------------
# 1. Preprocessing
# ----------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every object/categorical column in place, using
    sklearn's LabelEncoder instead of a manual factorization loop."""
    df = df.copy()
    encoder = LabelEncoder()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = encoder.fit_transform(df[col].astype(str))
    return df


def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Fill missing numeric values using sklearn's SimpleImputer.
    strategy in {"mean", "median", "most_frequent"}."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    imputer = SimpleImputer(strategy=strategy)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


# ----------------------------------------------------------------------
# 2. Distance metrics (vectorized: one call handles the whole train set)
# ----------------------------------------------------------------------
def _minkowski(train_matrix: np.ndarray, query: np.ndarray, p: float) -> np.ndarray:
    return np.sum(np.abs(train_matrix - query) ** p, axis=1) ** (1.0 / p)


def _chebyshev(train_matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.max(np.abs(train_matrix - query), axis=1)


DISTANCE_METRICS = {
    "euclidean": lambda A, q: _minkowski(A, q, 2),
    "manhattan": lambda A, q: _minkowski(A, q, 1),
    "chebyshev": _chebyshev,
}


def compute_distances(train_matrix: np.ndarray, query: np.ndarray,
                       metric: str = "euclidean") -> np.ndarray:
    if metric not in DISTANCE_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(DISTANCE_METRICS)}")
    return DISTANCE_METRICS[metric](train_matrix, query)


# ----------------------------------------------------------------------
# 3. Sorting algorithms (return indices that would sort `distances`)
# ----------------------------------------------------------------------
def merge_sort_indices(distances: np.ndarray) -> np.ndarray:
    pairs = list(enumerate(distances))

    def _merge_sort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = _merge_sort(arr[:mid])
        right = _merge_sort(arr[mid:])
        merged, i, j = [], 0, 0
        while i < len(left) and j < len(right):
            if left[i][1] <= right[j][1]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    sorted_pairs = _merge_sort(pairs)
    return np.array([idx for idx, _ in sorted_pairs])


def quick_sort_indices(distances: np.ndarray) -> np.ndarray:
    pairs = list(enumerate(distances))

    def _quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2][1]
        left = [p for p in arr if p[1] < pivot]
        mid = [p for p in arr if p[1] == pivot]
        right = [p for p in arr if p[1] > pivot]
        return _quick_sort(left) + mid + _quick_sort(right)

    sorted_pairs = _quick_sort(pairs)
    return np.array([idx for idx, _ in sorted_pairs])


def heap_sort_indices(distances: np.ndarray) -> np.ndarray:
    heap = [(d, i) for i, d in enumerate(distances)]
    heapq.heapify(heap)
    order = []
    while heap:
        d, i = heapq.heappop(heap)
        order.append(i)
    return np.array(order)


SORTERS = {
    "merge": merge_sort_indices,
    "quick": quick_sort_indices,
    "heap": heap_sort_indices,
    "numpy": lambda d: np.argsort(d, kind="stable"),
}


def get_sorted_indices(distances: np.ndarray, algorithm: str = "merge") -> np.ndarray:
    if algorithm not in SORTERS:
        raise ValueError(f"Unknown sort algorithm '{algorithm}'. Choose from {list(SORTERS)}")
    return SORTERS[algorithm](distances)


# ----------------------------------------------------------------------
# 4. Neighbor selection with boundary tie-breaking
# ----------------------------------------------------------------------
def get_k_neighbor_indices(distances: np.ndarray, k: int,
                            sort_algorithm: str = "merge") -> np.ndarray:
    order = get_sorted_indices(distances, sort_algorithm)
    k = min(k, len(order))
    boundary_distance = distances[order[k - 1]]
    # include every point exactly as close as the k-th neighbor
    tied_mask = np.isclose(distances, boundary_distance)
    within_k = set(order[:k].tolist())
    all_tied = set(np.where(tied_mask)[0].tolist())
    neighbor_idx = np.array(sorted(within_k | all_tied,
                                    key=lambda i: distances[i]))
    return neighbor_idx


# ----------------------------------------------------------------------
# 5. Voting (uniform or distance-weighted), with class-count tie handling
# ----------------------------------------------------------------------
def vote(neighbor_labels: np.ndarray, neighbor_distances: np.ndarray,
          weighted: bool = False):
    classes, counts = np.unique(neighbor_labels, return_counts=True)

    if weighted:
        safe_d = np.where(neighbor_distances == 0, 1e-12, neighbor_distances)
        w = 1.0 / safe_d
        scores = np.array([w[neighbor_labels == c].sum() for c in classes])
    else:
        scores = counts.astype(float)

    top = scores.max()
    winners = classes[np.isclose(scores, top)]
    if len(winners) == 1:
        return winners[0]

    # class-count tie -> break using inverse-distance weight among tied classes
    safe_d = np.where(neighbor_distances == 0, 1e-12, neighbor_distances)
    w = 1.0 / safe_d
    tie_scores = {c: w[neighbor_labels == c].sum() for c in winners}
    return max(tie_scores, key=tie_scores.get)


# ----------------------------------------------------------------------
# 6. Estimator (single class, scikit-learn-style API)
# ----------------------------------------------------------------------
class AIKNNClassifier(BaseEstimator, ClassifierMixin):
    """k-NN classifier supporting uniform or distance-weighted voting,
    configurable distance metric and sorting algorithm.

    Parameters
    ----------
    n_neighbors : int
    metric      : {"euclidean", "manhattan", "chebyshev"}
    weights     : {"uniform", "distance"}
    sort_algorithm : {"merge", "quick", "heap", "numpy"}
    """

    def __init__(self, n_neighbors: int = 3, metric: str = "euclidean",
                 weights: str = "uniform", sort_algorithm: str = "merge"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.sort_algorithm = sort_algorithm

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def _predict_one(self, x: np.ndarray):
        distances = compute_distances(self.X_train_, x, self.metric)
        idx = get_k_neighbor_indices(distances, self.n_neighbors, self.sort_algorithm)
        return vote(self.y_train_[idx], distances[idx],
                    weighted=(self.weights == "distance"))

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def score(self, X, y) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y)))
