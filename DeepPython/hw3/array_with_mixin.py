import numpy as np
from numbers import Number
from functools import reduce


class GetterSetterMixin:
    def __init__(self, data):
        self._data = np.asarray(data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data


class PrettyPrintMixin:
    def __str__(self):
        return '[\n' + reduce(lambda a, b: str(a) + ",\n" + str(b), self.data) + '\n]'


class SaveMixin:
    def save(self, save_path: str):
        with open(save_path, "w+") as file:
            file.write(str(self))


class Array(GetterSetterMixin, PrettyPrintMixin, SaveMixin, np.lib.mixins.NDArrayOperatorsMixin):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        data = []
        for input in inputs:
            if isinstance(input, (np.ndarray, Number)):
                data.append(input)
            elif isinstance(input, self.__class__):
                data.append(input.data)
            else:
                raise NotImplementedError()

        return self.__class__(getattr(ufunc, method)(*data, **kwargs))
