"""
performance_comparison.py
==========================
Lab 06 (23CSE301) - A3: performance comparison of three kNN implementations
on the fatigue-prediction project dataset (merged_dataset.csv, target =
high_fatigue):

  1. "Custom (yours)"   -> student_lab05_functions.MyKNN  (Lab 5, A7)
  2. "Scikit-learn"     -> sklearn.neighbors.KNeighborsClassifier
  3. "AI-generated"     -> ai_knn.AIKNNClassifier

For each: Accuracy, Precision, Recall, F1-score on a held-out test split,
and mean fit+predict wall-clock time over 10 runs.

GenAI Tool used: Claude (Anthropic), Sonnet model, via claude.ai chat
interface, dated 2026-09-22.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from student_lab05_functions import MyKNN
from ai_knn import AIKNNClassifier

N_RUNS = 10
K = 3
RANDOM_STATE = 42


def load_data(path="merged_dataset.csv"):
    data = pd.read_csv(path)
    data = data.drop(columns=["participant_id", "date"])
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(data[column].mean())
    X = data.drop(columns=["high_fatigue"])
    y = data["high_fatigue"]
    return X, y


def time_runs(fit_predict_fn, n_runs=N_RUNS):
    times = []
    preds = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        preds = fit_predict_fn()
        times.append(time.perf_counter() - t0)
    return preds, float(np.mean(times)), float(np.std(times))


def evaluate(y_true, y_pred, name, mean_time, std_time):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Avg time (s, 10 runs)": mean_time,
        "Std time (s)": std_time,
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    results = []

    # ---- 1. Custom kNN (student's own code, Lab 5 A7) ----
    def run_custom():
        model = MyKNN(k=K).fit(X_train, y_train)
        return model.predict(X_test)

    preds, mean_t, std_t = time_runs(run_custom)
    results.append(evaluate(y_test, preds, "Custom (yours)", mean_t, std_t))

    # ---- 2. Scikit-learn kNN ----
    def run_sklearn():
        model = KNeighborsClassifier(n_neighbors=K)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    preds, mean_t, std_t = time_runs(run_sklearn)
    results.append(evaluate(y_test, preds, "Scikit-learn", mean_t, std_t))

    # ---- 3. AI-generated kNN ----
    def run_ai():
        model = AIKNNClassifier(n_neighbors=K, metric="euclidean",
                                 weights="uniform", sort_algorithm="numpy")
        model.fit(X_train.values, y_train.values)
        return model.predict(X_test.values)

    preds, mean_t, std_t = time_runs(run_ai)
    results.append(evaluate(y_test, preds, "AI-generated", mean_t, std_t))

    results_df = pd.DataFrame(results).set_index("Model")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== k-NN Performance Comparison (k=3, test_size=0.3, "
          f"{N_RUNS} runs) ===\n")
    print(results_df.to_string())
    results_df.to_csv("knn_comparison_results.csv")
    print("\nSaved table to knn_comparison_results.csv")
    return results_df


if __name__ == "__main__":
    main()
