import numpy as np

from hash_matrix import MatrixH


if __name__ == "__main__":
    np.random.seed(0)

    a = np.random.randint(0, 10, (10, 10))
    b = d = np.random.randint(0, 10, (10, 10))
    c = np.random.randint(0, 10, (10, 10))

    a[0][0] = -c[0][0]

    matrix_a = MatrixH(a)
    matrix_b = MatrixH(b)
    matrix_c = MatrixH(c)
    matrix_d = MatrixH(d)

    assert hash(matrix_a) == hash(matrix_c) and matrix_a != matrix_c and matrix_b == matrix_d

    matrix_ab = matrix_a @ matrix_b
    matrix_cd = MatrixH(c @ d)
    cashed_matrix_cd = matrix_c @ matrix_d
    assert matrix_ab != matrix_cd and matrix_ab == cashed_matrix_cd

    with open("artifacts/A.txt", "w+") as file:
        file.write(str(matrix_a))

    with open("artifacts/B.txt", "w+") as file:
        file.write(str(matrix_b))

    with open("artifacts/C.txt", "w+") as file:
        file.write(str(matrix_c))

    with open("artifacts/D.txt", "w+") as file:
        file.write(str(matrix_d))

    with open("artifacts/AB.txt", "w+") as file:
        file.write(str(matrix_ab))

    with open("artifacts/CD.txt", "w+") as file:
        file.write(str(matrix_cd))

    with open("artifacts/hash.txt", "w+") as file:
        file.write(str(f"AB hash: {hash(matrix_ab)}\nCD hash: {hash(matrix_cd)}\n"))
