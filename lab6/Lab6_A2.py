"""
Lab 06 - A2
Unit test cases (generated with AI assistance - Claude, Anthropic) for the
modular functions of BOTH last week's (Lab 05, student-written: A1.py /
A7.py) and this week's (Lab 06, AI-generated: Lab6_A1.py) kNN pipelines.

pandas.read_csv is patched with a small synthetic DataFrame before the
target modules are imported, because both A1.py (Lab 05) and Lab6_A1.py
run their main pipeline at import time (Lab6_A1.py is guarded by
`if __name__ == "__main__":`, A1.py is not, so it will execute on import -
the patch makes that safe to run without needing the real dataset).

Run with:  python -m unittest Lab6_A2.py   (or)   python Lab6_A2.py
Place this file in the same folder as A1.py, A7.py (Lab 05) and
Lab6_A1.py (Lab 06).

GenAI tool used: Claude (Anthropic)
"""

import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Synthetic dataset used only to safely satisfy the module-level
# `pd.read_csv("merged_dataset.csv")` calls in the imported files.
# ----------------------------------------------------------------------
    # NOTE: kept purely numeric (besides participant_id/date, which every
    # Lab 05 script drops) because the student's own A3/A4/A7/A8/A9.py
    # never call encode_data() - they assume an already-numeric feature
    # matrix, matching the real merged_dataset.csv used in the project.
_SYNTH = pd.DataFrame({
    "participant_id": [f"P{i}" for i in range(20)],
    "date": pd.date_range("2024-01-01", periods=20).astype(str),
    "age": [25, 30, np.nan, 40, 22, 35, 28, np.nan, 31, 29,
            26, 33, 45, np.nan, 24, 38, 27, 41, 23, 36],
    "sleep_hours": [7, 6, 5, 8, np.nan, 6, 7, 5, 6, 8,
                    7, 6, 5, 8, 6, np.nan, 7, 5, 6, 8],
    "high_fatigue": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                      0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
})

with patch("pandas.read_csv", return_value=_SYNTH.copy()):
    import A1 as lab5_a1        # Lab 05 - student's own code
    import A7 as lab5_a7        # Lab 05 - student's own MyKNN
    import Lab6_A1 as lab6_a1   # Lab 06 - AI-generated code (main guarded, safe)


# ======================================================================
# Tests for LAST WEEK'S (Lab 05, student) modular functions -- A1.py
# ======================================================================
class TestLab05Preprocessing(unittest.TestCase):

    def test_encode_data_converts_object_columns_to_numeric(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c"], "num": [1, 2, 3, 4]})
        out = lab5_a1.encode_data(df.copy())
        self.assertTrue(np.issubdtype(out["cat"].dtype, np.number))
        self.assertEqual(out["num"].tolist(), [1, 2, 3, 4])

    def test_fill_missing_mean(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = lab5_a1.fill_missing(df.copy(), method="mean")
        self.assertFalse(out["x"].isnull().any())
        self.assertAlmostEqual(out["x"].iloc[1], 2.0)

    def test_fill_missing_median(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 100.0]})
        out = lab5_a1.fill_missing(df.copy(), method="median")
        self.assertAlmostEqual(out["x"].iloc[1], df["x"].median())

    def test_fill_missing_mode(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 1.0, 2.0]})
        out = lab5_a1.fill_missing(df.copy(), method="mode")
        self.assertEqual(out["x"].iloc[1], 1.0)

    def test_fill_missing_no_nulls_is_noop(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = lab5_a1.fill_missing(df.copy(), method="mean")
        self.assertEqual(out["x"].tolist(), [1.0, 2.0, 3.0])


class TestLab05Distance(unittest.TestCase):

    def test_euclidean_distance(self):
        d = lab5_a1.distance([0, 0], [3, 4], method="euclidean")
        self.assertAlmostEqual(d, 5.0)

    def test_manhattan_distance(self):
        d = lab5_a1.distance([0, 0], [3, 4], method="manhattan")
        self.assertAlmostEqual(d, 7.0)

    def test_zero_distance_identical_points(self):
        d = lab5_a1.distance([2, 2], [2, 2], method="euclidean")
        self.assertEqual(d, 0.0)


class TestLab05Sorting(unittest.TestCase):

    def setUp(self):
        self.unsorted = [[5, 0], [1, 1], [3, 0], [2, 1], [4, 0]]
        self.expected = [1, 2, 3, 4, 5]

    def test_bubble_sort(self):
        out = lab5_a1.bubble_sort([r[:] for r in self.unsorted])
        self.assertEqual([r[0] for r in out], self.expected)

    def test_selection_sort(self):
        out = lab5_a1.selection_sort([r[:] for r in self.unsorted])
        self.assertEqual([r[0] for r in out], self.expected)

    def test_insertion_sort(self):
        out = lab5_a1.insertion_sort([r[:] for r in self.unsorted])
        self.assertEqual([r[0] for r in out], self.expected)

    def test_sorts_already_sorted_input(self):
        already = [[1, 0], [2, 1], [3, 0]]
        out = lab5_a1.bubble_sort([r[:] for r in already])
        self.assertEqual([r[0] for r in out], [1, 2, 3])


class TestLab05NeighborsAndVoting(unittest.TestCase):

    def test_find_neighbors_returns_k_items(self):
        X_train = [[0, 0], [1, 1], [5, 5], [2, 2]]
        y_train = [0, 0, 1, 1]
        neighbors = lab5_a1.find_neighbors(X_train, y_train, [0, 0], k=2, sort_type="bubble")
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0][1], 0)  # closest point's label

    def test_find_class_majority_vote(self):
        neighbors = [[0.1, 0, 0], [0.2, 0, 1], [0.3, 1, 2]]
        self.assertEqual(lab5_a1.find_class(neighbors), 0)

    def test_find_class_tie_break_uses_closest(self):
        neighbors = [[0.1, 1, 0], [0.5, 0, 1]]
        self.assertEqual(lab5_a1.find_class(neighbors), 1)

    def test_weighted_class_closer_point_dominates(self):
        neighbors = [[0.001, 1, 0], [10.0, 0, 1]]
        self.assertEqual(lab5_a1.weighted_class(neighbors), 1)

    def test_weighted_class_zero_distance_handled(self):
        neighbors = [[0.0, 0, 0], [5.0, 1, 1]]
        self.assertEqual(lab5_a1.weighted_class(neighbors), 0)


class TestLab05MyKNN(unittest.TestCase):

    def test_fit_predict_score_shapes(self):
        X_train = [[0, 0], [1, 1], [8, 8], [9, 9]]
        y_train = pd.Series([0, 0, 1, 1])
        X_test = [[0, 1], [9, 8]]
        y_test = pd.Series([0, 1])

        model = lab5_a7.MyKNN(k=1).fit(X_train, y_train)
        preds = model.predict(X_test)
        self.assertEqual(len(preds), 2)
        acc = model.score(X_test, y_test)
        self.assertEqual(acc, 1.0)


# ======================================================================
# Tests for THIS WEEK'S (Lab 06, AI-generated) modular functions
# ======================================================================
class TestLab06Preprocessing(unittest.TestCase):

    def test_encode_categorical(self):
        df = pd.DataFrame({"cat": ["b", "a", "c"], "num": [1, 2, 3]})
        out = lab6_a1.DataPreprocessor.encode_categorical(df)
        # sorted-order mapping: a=0, b=1, c=2
        self.assertEqual(out["cat"].tolist(), [1, 0, 2])

    def test_impute_missing_mean(self):
        df = pd.DataFrame({"x": [2.0, np.nan, 4.0]})
        out = lab6_a1.DataPreprocessor.impute_missing(df, "mean")
        self.assertAlmostEqual(out["x"].iloc[1], 3.0)

    def test_impute_missing_invalid_strategy_raises(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        with self.assertRaises(ValueError):
            lab6_a1.DataPreprocessor.impute_missing(df, "bogus")


class TestLab06Distance(unittest.TestCase):

    def test_euclidean_vectorized(self):
        train = np.array([[0, 0], [3, 4]])
        out = lab6_a1.DistanceMetrics.compute(train, np.array([0, 0]), "euclidean")
        np.testing.assert_allclose(out, [0.0, 5.0])

    def test_manhattan_vectorized(self):
        train = np.array([[0, 0], [3, 4]])
        out = lab6_a1.DistanceMetrics.compute(train, np.array([0, 0]), "manhattan")
        np.testing.assert_allclose(out, [0.0, 7.0])

    def test_invalid_metric_raises(self):
        with self.assertRaises(ValueError):
            lab6_a1.DistanceMetrics.compute(np.array([[0, 0]]), np.array([0, 0]), "cosine")


class TestLab06Sorter(unittest.TestCase):

    def setUp(self):
        self.records = [(5, "a", 0), (1, "b", 1), (3, "c", 2), (2, "d", 3)]
        self.expected = [1, 2, 3, 5]

    def test_merge_sort(self):
        out = lab6_a1.Sorter.sort(list(self.records), "merge")
        self.assertEqual([r[0] for r in out], self.expected)

    def test_quick_sort(self):
        out = lab6_a1.Sorter.sort(list(self.records), "quick")
        self.assertEqual([r[0] for r in out], self.expected)

    def test_heap_sort(self):
        out = lab6_a1.Sorter.sort(list(self.records), "heap")
        self.assertEqual([r[0] for r in out], self.expected)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            lab6_a1.Sorter.sort(list(self.records), "bogosort")


class TestLab06KNNCustom(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[0, 0], [1, 1], [8, 8], [9, 9]])
        self.y_train = np.array([0, 0, 1, 1])
        self.X_test = np.array([[0, 1], [9, 8]])
        self.y_test = np.array([0, 1])

    def test_majority_vote_predictions_and_perfect_score(self):
        model = lab6_a1.KNNCustom(k=1, weighted=False).fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        np.testing.assert_array_equal(preds, self.y_test)
        self.assertEqual(model.score(self.X_test, self.y_test), 1.0)

    def test_weighted_vote_predictions(self):
        model = lab6_a1.KNNCustom(k=3, weighted=True).fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), 2)

    def test_different_sort_algorithms_agree(self):
        results = []
        for algo in ["merge", "quick", "heap"]:
            m = lab6_a1.KNNCustom(k=2, sort_algorithm=algo).fit(self.X_train, self.y_train)
            results.append(list(m.predict(self.X_test)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
