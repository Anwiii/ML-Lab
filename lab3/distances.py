# distances.py
# A4, A5, A6, A7 — Distance and Vector Operations

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski as scipy_minkowski


# ── A4: Minkowski Distance ─────────────────────────────────────

def minkowski_distance(u, v, p):
    """
    Generalized Minkowski distance between vectors u and v.
    p=1 -> Manhattan distance
    p=2 -> Euclidean distance
    """
    u = np.array(u, dtype=float)
    v = np.array(v, dtype=float)
    return np.sum(np.abs(u - v) ** p) ** (1 / p)


# ── A5: Plot distances for p=1 to 10 ──────────────────────────

def plot_minkowski_distances(u, v, save_path="minkowski_plot.png"):
    """
    Compute and plot Minkowski distance between u and v for p = 1 to 10.
    Returns list of distances.
    """
    p_values = list(range(1, 11))
    distances = [minkowski_distance(u, v, p) for p in p_values]

    plt.figure(figsize=(8, 4))
    plt.plot(p_values, distances, marker='o', color='steelblue', linewidth=2)
    plt.title("Minkowski Distance vs p")
    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.xticks(p_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return distances


# ── A6: Compare with scipy ─────────────────────────────────────

def compare_with_scipy(u, v, p=2):
    """
    Compare custom Minkowski distance with scipy's implementation.
    Returns (custom_result, scipy_result, match).
    """
    custom = minkowski_distance(u, v, p)
    scipy_result = scipy_minkowski(u, v, p)
    match = np.isclose(custom, scipy_result)
    return custom, scipy_result, match


# ── A7: Dot product, Norm, Cosine Similarity ───────────────────

def dot_product(a, b):
    """Dot product of two vectors a and b."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.sum(a * b))


def euclidean_norm(v):
    """Euclidean (L2) norm of vector v."""
    v = np.array(v, dtype=float)
    return float(np.sqrt(np.sum(v ** 2)))


def cosine_similarity(a, b):
    """Cosine similarity between vectors a and b."""
    return dot_product(a, b) / (euclidean_norm(a) * euclidean_norm(b))


def compare_vector_ops(a, b):
    """
    Compare custom dot product and norm with numpy equivalents.
    Returns a dict of results.
    """
    return {
        "dot_custom":    dot_product(a, b),
        "dot_numpy":     float(np.dot(a, b)),
        "norm_a_custom": euclidean_norm(a),
        "norm_a_numpy":  float(np.linalg.norm(a)),
        "norm_b_custom": euclidean_norm(b),
        "norm_b_numpy":  float(np.linalg.norm(b)),
        "cosine_similarity": cosine_similarity(a, b),
    }
