# statistics.py
# A8, A9, A10 — Mean, Variance, Std Dev, Histogram

import numpy as np
import matplotlib.pyplot as plt


# ── A8: Custom statistics ──────────────────────────────────────

def compute_mean(data):
    """
    Column-wise mean of a 2D matrix (rows = samples, cols = features).
    """
    data = np.array(data, dtype=float)
    return np.sum(data, axis=0) / data.shape[0]


def compute_variance(data):
    """
    Column-wise variance of a 2D matrix.
    """
    data = np.array(data, dtype=float)
    mean = compute_mean(data)
    return np.sum((data - mean) ** 2, axis=0) / data.shape[0]


def compute_std(data):
    """
    Column-wise standard deviation of a 2D matrix.
    """
    return np.sqrt(compute_variance(data))


def compute_dataset_stats(data):
    """
    Compute mean, variance, and std for a dataset matrix.
    Returns dict of arrays (one value per feature).
    """
    return {
        "mean":     compute_mean(data),
        "variance": compute_variance(data),
        "std":      compute_std(data),
    }


# ── A9: Compare with numpy ─────────────────────────────────────

def compare_with_numpy(data):
    """
    Compare custom statistics with numpy built-ins.
    Returns dict with custom and numpy values side by side.
    """
    data = np.array(data, dtype=float)
    return {
        "mean_custom": compute_mean(data),
        "mean_numpy":  np.mean(data, axis=0),
        "std_custom":  compute_std(data),
        "std_numpy":   np.std(data, axis=0),
    }


# ── A10: Histogram ─────────────────────────────────────────────

def plot_histogram(data, feature_name, bins=10, save_path=None):
    """
    Plot histogram of a 1D feature array.
    Returns (mean, variance) computed from raw data.
    """
    data = np.array(data, dtype=float)
    mean = float(compute_mean(data.reshape(-1, 1))[0])
    variance = float(compute_variance(data.reshape(-1, 1))[0])

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, color='steelblue', edgecolor='black', alpha=0.8)
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')
    plt.title(f"Histogram of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    path = save_path if save_path else f"histogram_{feature_name}.png"
    plt.savefig(path)
    plt.show()

    return mean, variance
