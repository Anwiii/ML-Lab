"""
test_lab06.py
=============
Lab 06 (23CSE301) - Unit tests for both last week's (Lab 5) modular
functions and this week's AI-generated modules (ai_knn.py).

GenAI Tool used to generate these test cases: Claude (Anthropic), Sonnet
model, via claude.ai chat interface, dated 2026-09-22.

Run with:
    python -m unittest test_lab06.py -v
"""

import unittest
import numpy as np
import pandas as pd

import student_lab05_functions as s
import ai_knn as ai


# ============================================================
# Lab 5 (student's own) function tests
# ============================================================
class TestStudentDistance(unittest.TestCase):
    def test_euclidean_known_value(self):
        self.assertAlmostEqual(s.distance([0, 0], [3, 4], "euclidean"), 5.0)

    def test_manhattan_known_value(self):
        self.assertEqual(s.distance([0, 0], [3, 4], "manhattan"), 7)

    def test_zero_distance_identical_points(self):
        self.assertEqual(s.distance([1, 2, 3], [1, 2, 3], "euclidean"), 0)


class TestStudentSorting(unittest.TestCase):
    def setUp(self):
        self.data = [[5, "a"], [1, "b"], [4, "c"], [2, "d"], [3, "e"]]

    def test_bubble_sort_orders_by_first_element(self):
        result = s.bubble_sort([row[:] for row in self.data])
        self.assertEqual([r[0] for r in result], [1, 2, 3, 4, 5])

    def test_selection_sort_orders_by_first_element(self):
        result = s.selection_sort([row[:] for row in self.data])
        self.assertEqual([r[0] for r in result], [1, 2, 3, 4, 5])

    def test_insertion_sort_orders_by_first_element(self):
        result = s.insertion_sort([row[:] for row in self.data])
        self.assertEqual([r[0] for r in result], [1, 2, 3, 4, 5])

    def test_all_three_sorts_agree(self):
        b = [r[0] for r in s.bubble_sort([row[:] for row in self.data])]
        sel = [r[0] for r in s.selection_sort([row[:] for row in self.data])]
        ins = [r[0] for r in s.insertion_sort([row[:] for row in self.data])]
        self.assertEqual(b, sel)
        self.assertEqual(sel, ins)


class TestStudentPreprocessing(unittest.TestCase):
    def test_fill_missing_mean(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = s.fill_missing(df.copy(), "mean")
        self.assertAlmostEqual(out["x"].iloc[1], 2.0)
        self.assertEqual(out["x"].isnull().sum(), 0)

    def test_fill_missing_median(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 100.0, 3.0]})
        out = s.fill_missing(df.copy(), "median")
        self.assertFalse(out["x"].isnull().any())

    def test_encode_data_object_column(self):
        df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
        out = s.encode_data(df.copy())
        self.assertTrue(pd.api.types.is_integer_dtype(out["cat"]))


class TestStudentFindNeighborsAndClass(unittest.TestCase):
    def setUp(self):
        self.X_train = [[0, 0], [1, 1], [5, 5], [6, 6]]
        self.y_train = [0, 0, 1, 1]

    def test_find_neighbors_returns_k_closest(self):
        neighbors = s.find_neighbors(self.X_train, self.y_train, [0.1, 0.1], k=2)
        labels = [n[1] for n in neighbors]
        self.assertEqual(sorted(labels), [0, 0])

    def test_find_class_majority_vote(self):
        neighbors = [[1.0, 0], [2.0, 0], [3.0, 1]]
        self.assertEqual(s.find_class(neighbors), 0)

    def test_find_class_tie_uses_nearest(self):
        neighbors = [[1.0, 1], [2.0, 0]]
        self.assertEqual(s.find_class(neighbors), 1)

    def test_weighted_class_prefers_closer_points(self):
        # one very close 0, one far 1 -> weighted vote should favour 0
        neighbors = [[0.01, 0], [10.0, 1]]
        self.assertEqual(s.weighted_class(neighbors), 0)


class TestStudentMyKNN(unittest.TestCase):
    def setUp(self):
        X = pd.DataFrame({"a": [0, 0, 10, 10], "b": [0, 1, 10, 11]})
        y = pd.Series([0, 0, 1, 1])
        self.X_train, self.y_train = X, y
        self.model = s.MyKNN(k=1).fit(X, y)

    def test_predict_matches_nearest_point(self):
        preds = self.model.predict(pd.DataFrame({"a": [0], "b": [0]}))
        self.assertEqual(preds[0], 0)

    def test_score_is_one_on_training_points_k1(self):
        acc = self.model.score(self.X_train, self.y_train)
        self.assertEqual(acc, 1.0)


class TestStudentWeightedKNN(unittest.TestCase):
    def test_weighted_knn_perfect_separation(self):
        X = pd.DataFrame({"a": [0, 0, 10, 10], "b": [0, 1, 10, 11]})
        y = pd.Series([0, 0, 1, 1])
        model = s.WeightedKNN(k=3).fit(X, y)
        acc = model.score(X, y)
        self.assertEqual(acc, 1.0)


# ============================================================
# Lab 6 (AI-generated) module tests
# ============================================================
class TestAIDistances(unittest.TestCase):
    def test_euclidean_matches_manual(self):
        train = np.array([[0.0, 0.0], [3.0, 4.0]])
        query = np.array([0.0, 0.0])
        d = ai.compute_distances(train, query, "euclidean")
        np.testing.assert_allclose(d, [0.0, 5.0])

    def test_manhattan_matches_manual(self):
        train = np.array([[0.0, 0.0], [3.0, 4.0]])
        query = np.array([0.0, 0.0])
        d = ai.compute_distances(train, query, "manhattan")
        np.testing.assert_allclose(d, [0.0, 7.0])

    def test_chebyshev_matches_manual(self):
        train = np.array([[0.0, 0.0], [3.0, 4.0]])
        query = np.array([0.0, 0.0])
        d = ai.compute_distances(train, query, "chebyshev")
        np.testing.assert_allclose(d, [0.0, 4.0])

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            ai.compute_distances(np.array([[0.0]]), np.array([0.0]), "bogus")


class TestAISorting(unittest.TestCase):
    def setUp(self):
        self.distances = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
        self.expected_order_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_merge_sort_indices(self):
        idx = ai.merge_sort_indices(self.distances)
        np.testing.assert_allclose(self.distances[idx], self.expected_order_values)

    def test_quick_sort_indices(self):
        idx = ai.quick_sort_indices(self.distances)
        np.testing.assert_allclose(self.distances[idx], self.expected_order_values)

    def test_heap_sort_indices(self):
        idx = ai.heap_sort_indices(self.distances)
        np.testing.assert_allclose(self.distances[idx], self.expected_order_values)

    def test_all_sorters_agree_with_numpy(self):
        for algo in ["merge", "quick", "heap", "numpy"]:
            idx = ai.get_sorted_indices(self.distances, algo)
            np.testing.assert_allclose(self.distances[idx], self.expected_order_values)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError):
            ai.get_sorted_indices(self.distances, "bogus")


class TestAINeighborsAndVoting(unittest.TestCase):
    def test_get_k_neighbor_indices_basic(self):
        distances = np.array([3.0, 1.0, 2.0, 10.0])
        idx = ai.get_k_neighbor_indices(distances, k=2)
        self.assertEqual(set(idx.tolist()), {1, 2})

    def test_get_k_neighbor_indices_includes_boundary_ties(self):
        distances = np.array([1.0, 2.0, 2.0, 5.0])
        idx = ai.get_k_neighbor_indices(distances, k=2)
        # both points at distance 2.0 tie for the 2nd slot -> both included
        self.assertEqual(set(idx.tolist()), {0, 1, 2})

    def test_vote_uniform_majority(self):
        labels = np.array([0, 0, 1])
        distances = np.array([1.0, 2.0, 3.0])
        self.assertEqual(ai.vote(labels, distances, weighted=False), 0)

    def test_vote_weighted_prefers_closer_class(self):
        labels = np.array([0, 1])
        distances = np.array([0.1, 10.0])
        self.assertEqual(ai.vote(labels, distances, weighted=True), 0)

    def test_vote_count_tie_breaks_by_distance(self):
        labels = np.array([0, 1])
        distances = np.array([1.0, 100.0])
        self.assertEqual(ai.vote(labels, distances, weighted=False), 0)


class TestAIKNNClassifier(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
        self.y = np.array([0, 0, 1, 1])

    def test_fit_returns_self(self):
        model = ai.AIKNNClassifier(n_neighbors=1)
        self.assertIs(model.fit(self.X, self.y), model)

    def test_predict_nearest_point_uniform(self):
        model = ai.AIKNNClassifier(n_neighbors=1, weights="uniform").fit(self.X, self.y)
        preds = model.predict(np.array([[0, 0], [10, 10.5]]))
        np.testing.assert_array_equal(preds, [0, 1])

    def test_predict_distance_weighted(self):
        model = ai.AIKNNClassifier(n_neighbors=3, weights="distance").fit(self.X, self.y)
        preds = model.predict(np.array([[0, 0.4]]))
        self.assertEqual(preds[0], 0)

    def test_score_perfect_separation(self):
        model = ai.AIKNNClassifier(n_neighbors=1).fit(self.X, self.y)
        self.assertEqual(model.score(self.X, self.y), 1.0)

    def test_all_metrics_run_without_error(self):
        for metric in ["euclidean", "manhattan", "chebyshev"]:
            model = ai.AIKNNClassifier(n_neighbors=2, metric=metric).fit(self.X, self.y)
            preds = model.predict(self.X)
            self.assertEqual(len(preds), len(self.y))

    def test_all_sort_algorithms_give_same_predictions(self):
        preds_by_algo = {}
        for algo in ["merge", "quick", "heap", "numpy"]:
            model = ai.AIKNNClassifier(n_neighbors=3, sort_algorithm=algo).fit(self.X, self.y)
            preds_by_algo[algo] = model.predict(self.X).tolist()
        first = next(iter(preds_by_algo.values()))
        for algo, preds in preds_by_algo.items():
            self.assertEqual(preds, first, msg=f"{algo} disagrees with merge sort")


class TestAIPreprocessing(unittest.TestCase):
    def test_impute_missing_mean(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        out = ai.impute_missing(df.copy(), "mean")
        self.assertAlmostEqual(out["x"].iloc[1], 2.0)

    def test_encode_categorical_object_column(self):
        df = pd.DataFrame({"cat": ["x", "y", "x"]})
        out = ai.encode_categorical(df.copy())
        self.assertTrue(pd.api.types.is_integer_dtype(out["cat"]) or
                         pd.api.types.is_numeric_dtype(out["cat"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
