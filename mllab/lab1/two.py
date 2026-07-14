# 2. Matrix Multiplication 
def get_matrix(rows, cols):
    matrix = []
    print("Enter the matrix:")
    for i in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)
    return matrix
def multiply(A, B, r1, c1, c2):
    result = []
    for i in range(r1):
        result.append([0] * c2)
    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] += A[i][k] * B[k][j]
    return result
def print_matrix(matrix):
    print("Product Matrix:")
    for row in matrix:
        print(row)
def main():
    r1 = int(input("Rows of Matrix A: "))
    c1 = int(input("Columns of Matrix A: "))
    r2 = int(input("Rows of Matrix B: "))
    c2 = int(input("Columns of Matrix B: "))
    if c1 != r2:
        print("Matrix multiplication is not possible.")
        return
    print("Matrix A")
    A = get_matrix(r1, c1)
    print("Matrix B")
    B = get_matrix(r2, c2)
    result = multiply(A, B, r1, c1, c2)
    print_matrix(result)
main()