# A2 - Label Encoding and One-Hot Encoding
# Simple version using plain loops, no numpy/pandas tricks


def label_encode(data):
    # find unique categories
    unique_values = []
    for item in data:
        if item not in unique_values:
            unique_values.append(item)
    unique_values.sort()

    # give each category a number
    encoded = []
    for item in data:
        for i in range(len(unique_values)):
            if item == unique_values[i]:
                encoded.append(i)
                break

    return encoded, unique_values


def one_hot_encode(data):
    # find unique categories
    unique_values = []
    for item in data:
        if item not in unique_values:
            unique_values.append(item)
    unique_values.sort()
    one_hot = []
    for item in data:
        row = [0] * len(unique_values)
        for i in range(len(unique_values)):
            if item == unique_values[i]:
                row[i] = 1
        one_hot.append(row)

    return one_hot, unique_values
if __name__ == "__main__":
    education_sample = ["Graduation", "PhD", "Master", "Graduation", "Basic"]
    encoded, categories = label_encode(education_sample)
    print("Label Encoding Example")
    print("Original :", education_sample)
    print("Categories found :", categories)
    print("Encoded  :", encoded)
    print()
    marital_sample = ["Single", "Married", "Single", "Divorced"]
    one_hot, categories2 = one_hot_encode(marital_sample)
    print("One-Hot Encoding Example")
    print("Original :", marital_sample)
    print("Categories found :", categories2)
    for row in one_hot:
        print(row)