"""
Lab 06 - A1
Repeat of Lab 05 experiments (A1-A9) using a GenAI tool (Claude, Anthropic)
for code generation. Function definitions / modularization plan are the
student's own; Claude was prompted to implement each module using a
different internal approach (class-based, vectorized with NumPy, and
different sorting algorithms) than the student's original Lab 05 code,
so the two can be meaningfully compared in Lab6_A3.py.

GenAI tool used for THIS ENTIRE FILE: Claude (Anthropic, Sonnet)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# ======================================================================
# A1(a) + A1(b): Preprocessing - Encoding & Missing-value Imputation
# GenAI tool used: Claude
# ======================================================================
class DataPreprocessor:
    """Handles categorical encoding and missing-value imputation."""

    @staticmethod
    def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """Label-encode every non-numeric column using a dict mapping
        built from the sorted unique values (deterministic, unlike
        pd.factorize's first-seen order)."""
        df = df.copy()
        for col in df.select_dtypes(include="object").columns:
            categories = sorted(df[col].dropna().unique())
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            df[col] = df[col].map(mapping)
        return df

    @staticmethod
    def impute_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        """Fill missing values column-wise using the chosen central
        tendency measure."""
        df = df.copy()
        strategies = {
            "mean": lambda s: s.mean(),
            "median": lambda s: s.median(),
            "mode": lambda s: s.mode().iloc[0] if not s.mode().empty else 0,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        for col in df.columns:
            if df[col].isna().any():
                fill_value = strategies[strategy](df[col])
                df[col] = df[col].fillna(fill_value)
        return df


# ======================================================================
# A1(c): Distance Calculation - vectorized with NumPy
# GenAI tool used: Claude
# ======================================================================
class DistanceMetrics:
    """Vectorized pairwise distance computation between a single test
    point and every row of a training matrix."""

    @staticmethod
    def compute(train_matrix: np.ndarray, point: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        diff = train_matrix - point
        if metric == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=1))
        elif metric == "manhattan":
            return np.sum(np.abs(diff), axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")


# ======================================================================
# A1(d): Sorting algorithms - 3 distinct DSA algorithms (config parameter)
# GenAI tool used: Claude
# ======================================================================
class Sorter:
    """Sorts a list of (distance, label, index) tuples by distance.
    Implements merge sort, quick sort and heap sort (different set
    from the bubble/selection/insertion trio used in the student's
    Lab 05 code) so the two files remain independently comparable.
    """

    @staticmethod
    def merge_sort(records):
        if len(records) <= 1:
            return records
        mid = len(records) // 2
        left = Sorter.merge_sort(records[:mid])
        right = Sorter.merge_sort(records[mid:])
        return Sorter._merge(left, right)

    @staticmethod
    def _merge(left, right):
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i][0] <= right[j][0]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged

    @staticmethod
    def quick_sort(records):
        if len(records) <= 1:
            return records
        pivot = records[len(records) // 2][0]
        lesser = [r for r in records if r[0] < pivot]
        equal = [r for r in records if r[0] == pivot]
        greater = [r for r in records if r[0] > pivot]
        return Sorter.quick_sort(lesser) + equal + Sorter.quick_sort(greater)

    @staticmethod
    def heap_sort(records):
        import heapq
        heap = list(records)
        heapq.heapify(heap)
        return [heapq.heappop(heap) for _ in range(len(heap))]

    @staticmethod
    def sort(records, algorithm="merge"):
        algorithms = {
            "merge": Sorter.merge_sort,
            "quick": Sorter.quick_sort,
            "heap": Sorter.heap_sort,
        }
        if algorithm not in algorithms:
            raise ValueError(f"Unknown sort algorithm: {algorithm}")
        return algorithms[algorithm](records)


# ======================================================================
# A1(e) + A1(f) + A2 (weighted) + A7: the classifier itself
# GenAI tool used: Claude
# ======================================================================
class KNNCustom:
    """Custom k-NN classifier supporting majority-vote and
    distance-weighted voting, configurable distance metric and
    sorting algorithm, with tie-breaking on both neighbor selection
    (implicit via stable sort) and class assignment."""

    def __init__(self, k=3, metric="euclidean", sort_algorithm="merge", weighted=False):
        self.k = k
        self.metric = metric
        self.sort_algorithm = sort_algorithm
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _neighbors_for_point(self, point):
        distances = DistanceMetrics.compute(self.X_train, point, self.metric)
        records = list(zip(distances.tolist(), self.y_train.tolist(), range(len(self.y_train))))
        records = Sorter.sort(records, self.sort_algorithm)
        return records[: self.k]

    def _vote(self, neighbors):
        if not self.weighted:
            # A1(f): majority voting with tie-break = closest neighbor's class
            votes = {}
            for dist, label, _ in neighbors:
                votes[label] = votes.get(label, 0) + 1
            best = max(votes.values())
            winners = [lbl for lbl, cnt in votes.items() if cnt == best]
            if len(winners) == 1:
                return winners[0]
            return neighbors[0][1]  # tie -> nearest neighbor's class
        else:
            # A2: weighted voting by inverse distance
            weights = {}
            for dist, label, _ in neighbors:
                w = 1e6 if dist == 0 else 1.0 / dist
                weights[label] = weights.get(label, 0.0) + w
            best = max(weights.values())
            winners = [lbl for lbl, wt in weights.items() if wt == best]
            if len(winners) == 1:
                return winners[0]
            return neighbors[0][1]

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        predictions = [self._vote(self._neighbors_for_point(pt)) for pt in X_test]
        return np.array(predictions)

    def score(self, X_test, y_test):
        y_test = np.asarray(y_test)
        preds = self.predict(X_test)
        return float(np.mean(preds == y_test))


# ======================================================================
# Main program - mirrors A1 through A9 of Lab 05, using the classes above
# GenAI tool used: Claude
# ======================================================================
if __name__ == "__main__":

    # ---- Load & preprocess (A1) ----
    raw = pd.read_csv("merged_dataset.csv")
    raw = raw.drop(columns=["participant_id", "date"])

    raw = DataPreprocessor.encode_categorical(raw)
    raw = DataPreprocessor.impute_missing(raw, strategy="mean")

    X = raw.drop(columns=["high_fatigue"])
    y = raw["high_fatigue"]

    print("A1: Number of rows:", len(X))
    print("A1: Number of features:", len(X.columns))

    # ---- A3: train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("A3: Training samples:", len(X_train))
    print("A3: Testing samples:", len(X_test))

    # ---- A4/A5/A6: sklearn kNN (k=3) ----
    sk_model = KNeighborsClassifier(n_neighbors=3)
    sk_model.fit(X_train, y_train)
    print("A4: sklearn model trained")
    print("A5: sklearn accuracy:", sk_model.score(X_test, y_test) * 100, "%")
    print("A6: sklearn predictions:", sk_model.predict(X_test))

    # ---- A7: custom kNN with fit/predict/score ----
    custom_model = KNNCustom(k=3, metric="euclidean", sort_algorithm="merge", weighted=False)
    custom_model.fit(X_train, y_train)
    print("A7: custom predictions:", custom_model.predict(X_test))
    print("A7: custom accuracy:", custom_model.score(X_test, y_test))

    # ---- A8: comparison across k, custom vs sklearn ----
    k_values = [1, 3, 5, 7, 9, 11]
    custom_test_acc, sklearn_test_acc = [], []
    custom_train_acc, sklearn_train_acc = [], []

    for k in k_values:
        cm = KNNCustom(k=k).fit(X_train, y_train)
        custom_test_acc.append(cm.score(X_test, y_test))
        custom_train_acc.append(cm.score(X_train, y_train))

        sm = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        sklearn_test_acc.append(sm.score(X_test, y_test))
        sklearn_train_acc.append(sm.score(X_train, y_train))

    print("A8: k values:", k_values)
    print("A8: custom test accuracy:", custom_test_acc)
    print("A8: sklearn test accuracy:", sklearn_test_acc)

    try:
        import matplotlib.pyplot as plt
        plt.plot(k_values, custom_test_acc, marker="o", label="Custom kNN (Claude-generated)")
        plt.plot(k_values, sklearn_test_acc, marker="o", label="Sklearn kNN")
        plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title("A8: Custom vs Sklearn kNN")
        plt.legend(); plt.savefig("A8_comparison.png")
        plt.close()
    except Exception as e:
        print("A8: plotting skipped:", e)

    # ---- A9: weighted kNN vs plain custom kNN ----
    weighted_test_acc, weighted_train_acc = [], []
    for k in k_values:
        wm = KNNCustom(k=k, weighted=True).fit(X_train, y_train)
        weighted_test_acc.append(wm.score(X_test, y_test))
        weighted_train_acc.append(wm.score(X_train, y_train))

    print("A9: weighted test accuracy:", weighted_test_acc)

    try:
        import matplotlib.pyplot as plt
        plt.plot(k_values, custom_test_acc, marker="o", label="Normal kNN")
        plt.plot(k_values, weighted_test_acc, marker="o", label="Weighted kNN")
        plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title("A9: Normal vs Weighted kNN")
        plt.legend(); plt.savefig("A9_comparison.png")
        plt.close()
    except Exception as e:
        print("A9: plotting skipped:", e)
