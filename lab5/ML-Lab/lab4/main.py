# main.py
# 23CSE301 — Lab Session 03
# Run this file to execute all experiments (A1 to A11)
# on both the marketing_campaign and PMData datasets.

import numpy as np
import pandas as pd

from encoding_funcs import label_encode, one_hot_encode, apply_encoding
from distances import (plot_minkowski_distances, compare_with_scipy,
                       compare_vector_ops)
from stats_funcs import compare_with_numpy, plot_histogram
from clustering import kmeans, compute_wcss, normalize

# ── Load datasets ──────────────────────────────────────────────

FILE = "Lab_Session_Data__1_.xlsx"

df_mc = pd.read_excel(FILE, sheet_name="marketing_campaign")
print(f"marketing_campaign loaded: {df_mc.shape}")

# For PMData: update this path to your actual file
# df_pm = pd.read_csv("pmdata.csv")
# For now we use Purchase data sheet as a stand-in demo
df_pm = pd.read_excel(FILE, sheet_name="Purchase data")
print(f"Project dataset loaded:    {df_pm.shape}")

# ══════════════════════════════════════════════════════════════
# MARKETING_CAMPAIGN DATASET
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MARKETING_CAMPAIGN DATASET")
print("=" * 60)

# ── A1: Datatype identification ────────────────────────────────
print("\nA1 — Feature Datatypes:")
datatype_map = {
    "ID": "Nominal", "Year_Birth": "Ratio", "Education": "Ordinal",
    "Marital_Status": "Nominal", "Income": "Ratio", "Kidhome": "Ratio",
    "Teenhome": "Ratio", "Dt_Customer": "Interval", "Recency": "Ratio",
    "MntWines": "Ratio", "MntFruits": "Ratio", "MntMeatProducts": "Ratio",
    "MntFishProducts": "Ratio", "MntSweetProducts": "Ratio",
    "MntGoldProds": "Ratio", "NumDealsPurchases": "Ratio",
    "NumWebPurchases": "Ratio", "NumCatalogPurchases": "Ratio",
    "NumStorePurchases": "Ratio", "NumWebVisitsMonth": "Ratio",
    "AcceptedCmp1": "Nominal", "AcceptedCmp2": "Nominal",
    "AcceptedCmp3": "Nominal", "AcceptedCmp4": "Nominal",
    "AcceptedCmp5": "Nominal", "Response": "Nominal",
    "Complain": "Nominal", "Z_CostContact": "Ratio", "Z_Revenue": "Ratio",
}
for col, dtype in datatype_map.items():
    print(f"  {col:25s} -> {dtype}")

# ── A2 & A3: Encoding ─────────────────────────────────────────
print("\nA2 & A3 — Encoding:")
df_enc, edu_map = apply_encoding(df_mc)
print(f"  Education label mapping : {edu_map}")
print(f"  Shape before encoding   : {df_mc.shape}")
print(f"  Shape after encoding    : {df_enc.shape}")
print(f"  Matrix rank             : {np.linalg.matrix_rank(df_enc.select_dtypes(include=[np.number]).values)}")

# ── Build numeric feature matrix ───────────────────────────────
feat_matrix = df_enc.select_dtypes(include=[np.number]).values
vec1 = feat_matrix[0]
vec2 = feat_matrix[1]

# ── A4 & A5: Minkowski distances ──────────────────────────────
print("\nA4 & A5 — Minkowski distance (p=1 to 10):")
dists = plot_minkowski_distances(vec1, vec2, save_path="mc_minkowski_plot.png")
for p, d in enumerate(dists, start=1):
    print(f"  p={p:2d}  ->  {d:.4f}")

# ── A6: Compare with scipy ────────────────────────────────────
print("\nA6 — scipy comparison (p=2):")
custom, scipy_val, match = compare_with_scipy(vec1, vec2, p=2)
print(f"  Custom : {custom:.10f}")
print(f"  Scipy  : {scipy_val:.10f}")
print(f"  Match  : {match}")

# ── A7: Dot product, norm, cosine similarity ──────────────────
print("\nA7 — Vector operations:")
ops = compare_vector_ops(vec1, vec2)
for key, val in ops.items():
    print(f"  {key:22s}: {val:.6f}")

# ── A8 & A9: Statistics ───────────────────────────────────────
print("\nA8 & A9 — Mean and Std (first 5 features):")
stats = compare_with_numpy(feat_matrix)
print("  Custom mean:", stats["mean_custom"][:5].round(4))
print("  NumPy  mean:", stats["mean_numpy"][:5].round(4))
print("  Custom std :", stats["std_custom"][:5].round(4))
print("  NumPy  std :", stats["std_numpy"][:5].round(4))

# ── A10: Histogram ────────────────────────────────────────────
print("\nA10 — Histogram (Income):")
income = df_enc["Income"].dropna().values
mean_inc, var_inc = plot_histogram(income, "Income", bins=10, save_path="mc_histogram_income.png")
print(f"  Mean     : {mean_inc:.2f}")
print(f"  Variance : {var_inc:.2f}")

# ── A11: K-Means ──────────────────────────────────────────────
print("\nA11 — K-Means (K=3):")
norm_matrix = normalize(feat_matrix)
labels, centroids, iters = kmeans(norm_matrix, k=3)
wcss = compute_wcss(norm_matrix, labels, centroids)
print(f"  Converged in {iters} iterations")
print(f"  WCSS: {wcss:.4f}")
for i in range(3):
    print(f"  Cluster {i}: {np.sum(labels == i)} points")

# ══════════════════════════════════════════════════════════════
# PROJECT DATASET (PMData / Purchase data)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PROJECT DATASET")
print("=" * 60)

# ── A2 & A3: Encoding ─────────────────────────────────────────
print("\nA2 & A3 — Encoding:")
df_pm_enc = df_pm.copy()

# Keep only the 4 core numeric columns (the rest have NaNs or are derived)
core_cols = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)"]
df_pm_enc = df_pm_enc[core_cols].dropna()

print(f"  Shape before encoding: {df_pm.shape}")
print(f"  All columns numeric — no encoding needed")
print(f"  Shape after  encoding: {df_pm_enc.shape}")

pm_matrix = df_pm_enc.select_dtypes(include=[np.number]).values
pm_vec1 = pm_matrix[0]
pm_vec2 = pm_matrix[1]

try:
    rank = np.linalg.matrix_rank(pm_matrix)
except np.linalg.LinAlgError:
    rank = "N/A (dataset too small for SVD)"
print(f"  Matrix rank: {rank}")

# ── A4 & A5: Minkowski distances ──────────────────────────────
print("\nA4 & A5 — Minkowski distance (p=1 to 10):")
pm_dists = plot_minkowski_distances(pm_vec1, pm_vec2, save_path="pm_minkowski_plot.png")
for p, d in enumerate(pm_dists, start=1):
    print(f"  p={p:2d}  ->  {d:.4f}")

# ── A6: Compare with scipy ────────────────────────────────────
print("\nA6 — scipy comparison (p=2):")
c, s, m = compare_with_scipy(pm_vec1, pm_vec2, p=2)
print(f"  Custom : {c:.10f}")
print(f"  Scipy  : {s:.10f}")
print(f"  Match  : {m}")

# ── A7: Vector operations ─────────────────────────────────────
print("\nA7 — Vector operations:")
pm_ops = compare_vector_ops(pm_vec1, pm_vec2)
for key, val in pm_ops.items():
    print(f"  {key:22s}: {val:.6f}")

# ── A8 & A9: Statistics ───────────────────────────────────────
print("\nA8 & A9 — Mean and Std:")
pm_stats = compare_with_numpy(pm_matrix)
print("  Custom mean:", pm_stats["mean_custom"].round(4))
print("  NumPy  mean:", pm_stats["mean_numpy"].round(4))
print("  Custom std :", pm_stats["std_custom"].round(4))
print("  NumPy  std :", pm_stats["std_numpy"].round(4))

# ── A10: Histogram ────────────────────────────────────────────
print("\nA10 — Histogram (Payment (Rs)):")
payment = df_pm_enc["Payment (Rs)"].values
mean_pay, var_pay = plot_histogram(payment, "Payment_Rs", bins=5, save_path="pm_histogram_payment.png")
print(f"  Mean     : {mean_pay:.2f}")
print(f"  Variance : {var_pay:.2f}")

# ── A11: K-Means ──────────────────────────────────────────────
K = 2
print(f"\nA11 — K-Means (K={K}):")
pm_norm = normalize(pm_matrix)
pm_labels, pm_centroids, pm_iters = kmeans(pm_norm, k=K)
pm_wcss = compute_wcss(pm_norm, pm_labels, pm_centroids)
print(f"  Converged in {pm_iters} iterations")
print(f"  WCSS: {pm_wcss:.4f}")
for i in range(K):
    print(f"  Cluster {i}: {np.sum(pm_labels == i)} points")

print("\nAll experiments complete.")
