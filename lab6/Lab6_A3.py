"""
Lab 06 - A3
Performance comparison of 3 versions of the k-NN algorithm on the project
data (merged_dataset.csv):
    1. Own code            -> MyKNN from the student's Lab 05 A7.py
    2. Scikit-Learn         -> sklearn.neighbors.KNeighborsClassifier
    3. GenAI-generated code -> KNNCustom from Lab6_A1.py (this week, Claude)

Metrics reported: Accuracy, Precision, Recall, F1-score, and average
prediction time over 10 runs.

Place this file alongside A7.py (Lab 05) and Lab6_A1.py (Lab 06) and run
it from the folder containing merged_dataset.csv.

GenAI tool used: Claude (Anthropic)
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from A7 import MyKNN                 # Lab 05 - student's own code
from Lab6_A1 import KNNCustom        # Lab 06 - GenAI-generated code

N_RUNS = 10
K = 3


def load_data(path="merged_dataset.csv"):
    data = pd.read_csv(path)
    data = data.drop(columns=["participant_id", "date"])
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(data[column].mean())
    X = data.drop(columns=["high_fatigue"])
    y = data["high_fatigue"]
    return X, y


def time_predict(fit_fn, predict_fn, n_runs=N_RUNS):
    """Fit once, then time `predict_fn` averaged over n_runs calls."""
    fit_fn()
    total = 0.0
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn()
        total += time.perf_counter() - start
    return total / n_runs


def evaluate(name, y_true, y_pred, avg_time_sec):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Avg Prediction Time (s, over {} runs)".format(N_RUNS): avg_time_sec,
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    results = []

    # ---- 1. Own code (Lab 05, MyKNN) ----
    own_model = MyKNN(k=K)
    avg_time = time_predict(
        fit_fn=lambda: own_model.fit(X_train, y_train),
        predict_fn=lambda: own_model.predict(X_test),
    )
    own_preds = own_model.predict(X_test)
    results.append(evaluate("Own code (Lab05 MyKNN)", y_test, own_preds, avg_time))

    # ---- 2. Scikit-Learn ----
    sk_model = KNeighborsClassifier(n_neighbors=K)
    avg_time = time_predict(
        fit_fn=lambda: sk_model.fit(X_train, y_train),
        predict_fn=lambda: sk_model.predict(X_test),
    )
    sk_preds = sk_model.predict(X_test)
    results.append(evaluate("Scikit-Learn KNeighborsClassifier", y_test, sk_preds, avg_time))

    # ---- 3. GenAI-generated code (Lab 06, KNNCustom) ----
    ai_model = KNNCustom(k=K, metric="euclidean", sort_algorithm="merge", weighted=False)
    avg_time = time_predict(
        fit_fn=lambda: ai_model.fit(X_train, y_train),
        predict_fn=lambda: ai_model.predict(X_test),
    )
    ai_preds = ai_model.predict(X_test)
    results.append(evaluate("GenAI-generated (Claude, Lab6_A1 KNNCustom)", y_test, ai_preds, avg_time))

    report = pd.DataFrame(results).set_index("Model")
    pd.set_option("display.width", 120)
    print(report.to_string(float_format=lambda v: f"{v:.4f}" if abs(v) < 1 else f"{v:.6f}"))

    report.to_csv("A3_performance_comparison.csv")
    print("\nSaved table to A3_performance_comparison.csv")


if __name__ == "__main__":
    main()
