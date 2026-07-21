"""
A2. IRCTC Stock Price Analysis
Load "IRCTC Stock Price" worksheet from Lab Session Data.xlsx and analyze.
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------------
file_path = "Lab Session Data.xlsx"          # update path if needed
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

price = df["Price"].to_numpy(dtype=float)     # Column D
chg_pct = df["Chg%"].to_numpy(dtype=float)     # Column I
day = df["Day"].to_numpy()
month = df["Month"].to_numpy()

# ------------------------------------------------------------------
# 2. Mean and Variance using numpy
# ------------------------------------------------------------------
mean_np = np.mean(price)
var_np = np.var(price)

print("=== Mean & Variance (NumPy) ===")
print(f"Mean (numpy)     : {mean_np:.4f}")
print(f"Variance (numpy) : {var_np:.4f}")

# ------------------------------------------------------------------
# 3. Own functions for mean and variance
# ------------------------------------------------------------------
def my_mean(data):
    total = 0.0
    for x in data:
        total += x
    return total / len(data)

def my_variance(data):
    m = my_mean(data)
    total = 0.0
    for x in data:
        total += (x - m) ** 2
    return total / len(data)          # population variance (matches np.var default, ddof=0)

mean_custom = my_mean(price)
var_custom = my_variance(price)

print("\n=== Mean & Variance (Custom functions) ===")
print(f"Mean (custom)     : {mean_custom:.4f}")
print(f"Variance (custom) : {var_custom:.4f}")

print("\n=== Accuracy Comparison ===")
print(f"Mean difference     : {abs(mean_np - mean_custom):.10f}")
print(f"Variance difference : {abs(var_np - var_custom):.10f}")

# ------------------------------------------------------------------
# 4. Computational complexity comparison (10 runs, average time)
# ------------------------------------------------------------------
N_RUNS = 10

def time_function(func, *args):
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)

avg_time_np_mean = time_function(np.mean, price)
avg_time_custom_mean = time_function(my_mean, price)
avg_time_np_var = time_function(np.var, price)
avg_time_custom_var = time_function(my_variance, price)

print("\n=== Computational Complexity (avg over 10 runs) ===")
print(f"np.mean()   avg time : {avg_time_np_mean:.8f} s")
print(f"my_mean()   avg time : {avg_time_custom_mean:.8f} s")
print(f"np.var()    avg time : {avg_time_np_var:.8f} s")
print(f"my_variance() avg time : {avg_time_custom_var:.8f} s")

# ------------------------------------------------------------------
# 5. Wednesday price - sample mean vs population mean
# ------------------------------------------------------------------
wed_mask = (day == "Wed")
wed_prices = price[wed_mask]
wed_mean = np.mean(wed_prices)

print("\n=== Wednesday Price Analysis ===")
print(f"Population mean (all days) : {mean_np:.4f}")
print(f"Sample mean (Wednesdays)   : {wed_mean:.4f}")
print(f"Difference                 : {wed_mean - mean_np:.4f}")
if wed_mean > mean_np:
    print("Observation: Wednesday mean price is HIGHER than the population mean.")
else:
    print("Observation: Wednesday mean price is LOWER than the population mean.")

# ------------------------------------------------------------------
# 6. April price - sample mean vs population mean
# ------------------------------------------------------------------
apr_mask = (month == "Apr")
apr_prices = price[apr_mask]
apr_mean = np.mean(apr_prices)

print("\n=== April Price Analysis ===")
print(f"Population mean (all months) : {mean_np:.4f}")
print(f"Sample mean (April)          : {apr_mean:.4f}")
print(f"Difference                    : {apr_mean - mean_np:.4f}")
if apr_mean > mean_np:
    print("Observation: April mean price is HIGHER than the population mean.")
else:
    print("Observation: April mean price is LOWER than the population mean.")

# ------------------------------------------------------------------
# 7. Probability of making a loss over the stock (from Chg%)
# ------------------------------------------------------------------
is_loss = lambda x: x < 0
loss_days = np.array([is_loss(x) for x in chg_pct])
p_loss = np.sum(loss_days) / len(chg_pct)

print("\n=== Probability of Loss ===")
print(f"P(Loss) = {p_loss:.4f}")

# ------------------------------------------------------------------
# 8. Probability of making a profit specifically on Wednesday
# ------------------------------------------------------------------
is_profit = lambda x: x > 0
wed_chg = chg_pct[wed_mask]
profit_on_wed = np.array([is_profit(x) for x in wed_chg])
p_profit_wed = np.sum(profit_on_wed) / len(chg_pct)   # P(Profit AND Wednesday)

print("\n=== Probability of Profit on Wednesday ===")
print(f"P(Profit ∩ Wednesday) = {p_profit_wed:.4f}")

# ------------------------------------------------------------------
# 9. Conditional probability P(Profit | Wednesday)
# ------------------------------------------------------------------
p_profit_given_wed = np.sum(profit_on_wed) / len(wed_chg)

print("\n=== Conditional Probability P(Profit | Wednesday) ===")
print(f"P(Profit | Wednesday) = {p_profit_given_wed:.4f}")

# ------------------------------------------------------------------
# 10. Scatter plot: Chg% vs Day of the week
# ------------------------------------------------------------------
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
day_to_num = {d: i for i, d in enumerate(day_order)}
day_numeric = np.array([day_to_num.get(d, -1) for d in day])

plt.figure(figsize=(8, 5))
plt.scatter(day_numeric, chg_pct, alpha=0.6, color="teal")
plt.xticks(range(len(day_order)), day_order)
plt.xlabel("Day of the Week")
plt.ylabel("Chg%")
plt.title("Chg% vs Day of the Week")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chg_vs_day_scatter.png", dpi=150)
print("\nScatter plot saved as 'chg_vs_day_scatter.png'")
plt.show()