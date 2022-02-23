import numpy as np

from array_with_mixin import Array


if __name__ == "__main__":
    np.random.seed(0)

    np_array_1 = np.random.randint(0, 10, (10, 10))
    np_array_2 = np.random.randint(0, 10, (10, 10))

    array_1 = Array(np_array_1)
    array_2 = Array(np_array_2)

    array_add = array_1 + array_2
    array_mul = array_1 * array_2
    array_matmul = array_1 @ array_2

    assert array_add == Array(np_array_1 + np_array_2)
    array_add.save("artifacts/array+.txt")

    assert array_mul == Array(np_array_1 * np_array_2)
    array_add.save("artifacts/array*.txt")

    assert array_matmul == Array(np_array_1 @ np_array_2)
    array_add.save("artifacts/array@.txt")
