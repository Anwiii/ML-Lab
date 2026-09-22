#4. Matrix Transpose

def get_matrix(rows, cols):
    matrix = []
    print("Enter the matrix:")
    for i in range(rows):
        row = list(map(int, input().split()))
        matrix.append(row)
    return matrix

def transpose(matrix, rows, cols):
    result = []

    for i in range(cols):
        row = []
        for j in range(rows):
            row.append(matrix[j][i])
        result.append(row)

    return result

def print_matrix(matrix):
    for row in matrix:
        print(row)

def main():
    rows = int(input("Enter number of rows: "))
    cols = int(input("Enter number of columns: "))

    matrix = get_matrix(rows, cols)

    trans = transpose(matrix, rows, cols)

    print("Transpose Matrix:")
    print_matrix(trans)

main()