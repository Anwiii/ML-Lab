def minkowski_distance(vector1, vector2, p):
    total = 0
    for i in range(len(vector1)):
        diff = abs(vector1[i] - vector2[i])
        total = total + diff ** p
    distance = total ** (1 / p)
    return distance


if __name__ == "__main__":
    v1 = [2, 4, 6]
    v2 = [1, 2, 3]

    print("Vector 1:", v1)
    print("Vector 2:", v2)
    print("Manhattan distance (p=1):", minkowski_distance(v1, v2, 1))
    print("Euclidean distance (p=2):", minkowski_distance(v1, v2, 2))
    print("Minkowski distance (p=5):", minkowski_distance(v1, v2, 5))