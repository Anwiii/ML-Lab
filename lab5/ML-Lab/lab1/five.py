# 5. Random Numbers - Mean, Median, Mode
import random, statistics

def generate_numbers(n=100, lo=100, hi=150):
    return [random.randint(lo, hi) for _ in range(n)]

def calc_mean(nums):
    return statistics.mean(nums)

def calc_median(nums):
    return statistics.median(nums)

def calc_mode(nums):
    return statistics.mode(nums)

def display_stats(nums):
    print(f"Mean:   {calc_mean(nums):.2f}")
    print(f"Median: {calc_median(nums)}")
    print(f"Mode:   {calc_mode(nums)}")

def main():
    nums = generate_numbers()
    display_stats(nums)

main()