# A8

def calculate_mean(data):
    num_rows = len(data)
    num_cols = len(data[0])
    mean_values = []

    for col in range(num_cols):
        total = 0
        for row in range(num_rows):
            total = total + data[row][col]
        mean_values.append(total / num_rows)

    return mean_values
def calculate_variance(data):
    mean_values = calculate_mean(data)
    num_rows = len(data)
    num_cols = len(data[0])
    variance_values = []

    for col in range(num_cols):
        total = 0
        for row in range(num_rows):
            total = total + (data[row][col] - mean_values[col]) ** 2
        variance_values.append(total / num_rows)

    return variance_values


def calculate_std(data):
    variance_values = calculate_variance(data)
    std_values = []
    for v in variance_values:
        std_values.append(v ** 0.5)
    return std_values


if __name__ == "__main__":
    sample_data = [[2, 4], [4, 8], [6, 12]]
    print("Sample data:", sample_data)
    print("Mean    :", calculate_mean(sample_data))
    print("Variance:", calculate_variance(sample_data))
    print("Std Dev :", calculate_std(sample_data))