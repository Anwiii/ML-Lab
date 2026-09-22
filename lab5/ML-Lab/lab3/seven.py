import numpy as np
def dot_product(vector1, vector2):
    total = 0
    for i in range(len(vector1)):
        total = total + vector1[i] * vector2[i]
    return total
def vector_length(vector):
    total = 0
    for value in vector:
        total = total + value ** 2
    length = total ** 0.5
    return length   
if __name__ == "__main__":
    v1 = [3, 4]
    v2 = [1, 2]

    my_dot = dot_product(v1, v2)
    numpy_dot = np.dot(v1, v2)

    my_length1 = vector_length(v1)
    numpy_length1 = np.linalg.norm(v1)

    print("My dot product   :", my_dot)
    print("Numpy dot product:", numpy_dot)
    print()
    print("My vector length (v1)   :", my_length1)
    print("Numpy vector length (v1):", numpy_length1)