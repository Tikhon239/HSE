from matrix import Matrix


class HashMixin:
    def __hash__(self):
        """
        Abs first element
        """
        return abs(int(self.data[0][0]))


class MatrixH(HashMixin, Matrix):
    matmul_cash = {}

    def __matmul__(self, other):
        self._shape_eq(other, True)

        key = (self.__hash__(), other.__hash__())
        if key not in self.__class__.matmul_cash:
            self.__class__.matmul_cash[key] = super(MatrixH, self).__matmul__(other)
        return self.__class__.matmul_cash[key]
