import numpy as np

from matrix import Matrix

if __name__ == "__main__":
    np.random.seed(0)

    array_1 = np.random.randint(0, 10, (10, 10))
    array_2 = np.random.randint(0, 10, (10, 10))

    matrix_1 = Matrix(array_1)
    matrix_2 = Matrix(array_2)

    matrix_add = matrix_1 + matrix_2
    matrix_mul = matrix_1 * matrix_2
    matrix_matmul = matrix_1 @ matrix_2

    assert matrix_add == Matrix(array_1 + array_2)
    with open("artifacts/matrix+.txt", "w+") as file:
        file.write(str(matrix_add))

    assert matrix_mul == Matrix(array_1 * array_2)
    with open("artifacts/matrix*.txt", "w+") as file:
        file.write(str(matrix_mul))

    assert matrix_matmul == Matrix(array_1 @ array_2)
    with open("artifacts/matrix@.txt", "w+") as file:
        file.write(str(matrix_matmul))
