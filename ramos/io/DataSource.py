from abc import abstractmethod
from copy import copy
import numpy as np

from ramos.utils.matrix import MatrixBuilder


__all__ = ['DataSource']


class Field:

    def __init__(self, name, ncomps, size, **kwargs):
        self.name = name
        self.ncomps = ncomps
        self.size = size
        self.__dict__.update(kwargs)


class DataSource:

    def __init__(self, pardim, ntimes):
        self.pardim = pardim
        self.ntimes = ntimes
        self._fields = {}
        self._mass = {}

    def clone(self, clear_cache=False):
        c = copy(self)
        if clear_cache:
            c._mass = {}
        return c

    def add_field(self, name, *args, **kwargs):
        self._fields[name] = Field(name, *args, **kwargs)

    def fields(self):
        yield from self._fields.values()

    def field(self, name):
        return self._fields[name]

    def levels(self):
        yield from range(0, self.ntimes)

    def mass_matrix(self, fields):
        if isinstance(fields, str):
            fields = [fields]
        builder = MatrixBuilder()
        for name in fields:
            if isinstance(name, tuple):
                name, scale = name
            else:
                scale = 1.0
            field = self.field(name)
            if name not in self._mass:
                self._mass[name] = self.field_mass_matrix(field)
            d, r, c = self._mass[name]
            print(d.shape, r.shape, c.shape, min(r), max(r), min(c), max(c))
            builder.add(*self._mass[name], field.ncomps, scale)
        return builder.build()

    def coefficients(self, fields, level=0, flatten=True):
        array = np.hstack([
            self.field_coefficients(self.field(name), level)
            for name in fields
        ])
        if flatten:
            return np.reshape(array, (np.prod(array.shape),))
        return array

    @abstractmethod
    def field_mass_matrix(self, field):
        raise NotImplementedError

    @abstractmethod
    def field_coefficients(self, field, level=0):
        raise NotImplementedError

    def __str__(self):
        return '{}(ntimes={}, pardim={}, fields=[{}])'.format(
            self.__class__.__name__,
            self.ntimes,
            self.pardim,
            ','.join('{}({},{})'.format(f.name, f.size, f.ncomps)
                     for f in self.fields()),
        )
