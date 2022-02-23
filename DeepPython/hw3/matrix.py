from typing import List, Tuple, Union
import numpy as np
from functools import reduce

class Matrix:
    def __init__(self, data: Union[List, np.ndarray]):
        self.data = data

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @data.setter
    def data(self, data):
        self._sequences_eq(data)
        self._data = data
        self._shape = (len(data), len(data[0]))

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if isinstance(index, int):
            return self._data[index]
        if isinstance(index, tuple):
            i, j = index
            if isinstance(i, int) and isinstance(j, int):
                return self._data[i][j]

        raise IndexError('only integers, tuple integers')

    def _type_eq(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Objects of different types')

    @staticmethod
    def _sequences_eq(data: Union[List, np.ndarray]):
        for row in data:
            if len(row) != len(data):
                raise ValueError('Creating matrix from ragged nested sequences')

    def _shape_eq(self, other, matmul: bool = False):
        self._type_eq(other)

        if matmul:
            if self.shape[1] != other.shape[0]:
                raise ValueError('Matrices must be the same shape')
        else:
            if self.shape != other.shape:
                raise ValueError('Matrices must be the same shape')

    def __eq__(self, other):
        self._type_eq(other)

        if self.shape != other.shape:
            return False
        # reduce(lambda row, tail: reduce(lambda a, b: a[0] == a[1] and b, zip(row[0], row[1])) and tail, zip(matrix_1, matrix_2))
        flag = True
        for row_1, row_2 in zip(self._data, other._data):
            for el_1, el_2 in zip(row_1, row_2):
                flag = flag and el_1 == el_2

        return flag

    def __add__(self, other):
        self._shape_eq(other)

        return self.__class__([
            [self[i, j] + other[i, j] for j in range(self.shape[1])] for i in range(self.shape[0])
        ])

    def __mul__(self, other):
        self._shape_eq(other)

        return self.__class__([
            [self[i, j] * other[i, j] for j in range(self.shape[1])] for i in range(self.shape[0])
        ])

    def __matmul__(self, other):
        self._shape_eq(other, True)

        return self.__class__([
                [sum([self[i, k] * other[k, j] for k in range(self.shape[1])]) for j in range(other.shape[1])]
                for i in range(self.shape[0])
        ])

    def __str__(self):
        return '[\n' + reduce(lambda a, b: str(a) + ",\n" + str(b), self._data) + '\n]'
