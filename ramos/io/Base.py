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
    """The class that all types of data sources should derive from. This class
    defines the minimal interface necessary for a new data source to work.
    """

    def __init__(self, pardim, ntimes):
        """Initialize a data source with a given number of parametric dimensions
        and time levels.

        This constructor should not be called directly, but as part of
        initialization code in subclasses.
        """
        self.pardim = pardim
        self.ntimes = ntimes
        self._fields = {}       # Map field names to Field objects
        self._mass = {}         # Mass matrix cache

    def clone(self, clear_cache=False):
        """Create a copy of this data source. If clear_cache is true, the copy receives
        a new mass matrix cache not shared with the original object.
        """
        c = copy(self)
        if clear_cache:
            c._mass = {}
        return c

    def add_field(self, name, *args, **kwargs):
        """add_field(name, ncomps, size, **kwargs)

        Add a field to this data source, with the given parameters.
        """
        self._fields[name] = Field(name, *args, **kwargs)

    def fields(self):
        """Iterate over all fields."""
        yield from self._fields.values()

    def field(self, name):
        """Get a field by name."""
        return self._fields[name]

    def levels(self):
        """Iterate over all time levels (by index, not by actual time)."""
        yield from range(0, self.ntimes)

    def mass_matrix(self, fields, single=False):
        """Compute the mass matrix for the given field(s). `fields` must be a single
        field, a list of fields, or a list of (field, scale), where `scale` is
        an optional scaling factor.
        """
        if isinstance(fields, str):
            fields = [fields]

        builder = MatrixBuilder()
        for name in fields:
            if isinstance(name, tuple):
                name, scale = name
            else:
                scale = 1.0
            field = self.field(name)

            # Check in the cache if this mass matrix has been computed before.
            # If not, compute it.
            if name not in self._mass:
                self._mass[name] = self.field_mass_matrix(field)

            # The cache contains the arguments that the matrix builder needs
            # to construct the matrix (see MatrixBuilder for more info)
            args = list(self._mass[name])
            args.append(1 if single else field.ncomps)
            args.append(scale)
            builder.add(*args)

        return builder.build()

    def unity_coefficients(self, fields, field, comp=0):
        """Return the unity coefficient vector for a field/component.

        The unity vector has ones for all degrees of freedom corresponding to
        that field/component, and zeros elsewhere.
        """
        coeffs = []
        for ff in fields:
            ff = self.field(ff)
            array = np.zeros((ff.size * ff.ncomps,))
            if ff.name == field:
                array[comp::ff.ncomps] = 1.0
            coeffs.append(array)
        return np.hstack(coeffs)

    def coefficients(self, fields, level=0, flatten=True):
        """Return the coefficient vector for one or more fields at a given time level.

        `fields` must be a field name or a list of fields. If `flatten` is
        false, a matrix of shape npts × ncomps is returned, otherwise the
        result is flattened to one dimension. (For multiple fields, `flatten`
        must be true.)
        """
        if isinstance(fields, str):
            field = self.field(fields)
            coeffs = self.field_coefficients(field, level)
            if flatten:
                return np.ndarray.flatten(coeffs)
            return np.reshape(coeffs, (field.size, field.ncomps))
        if not flatten:
            raise ValueError
        return np.hstack([
            np.ndarray.flatten(self.field_coefficients(self.field(name), level))
            for name in fields
        ])

    @abstractmethod
    def field_mass_matrix(self, field):
        """Return the mass matrix for a single field.

        Must be implemented in child classes."""
        raise NotImplementedError

    @abstractmethod
    def field_coefficients(self, field, level=0):
        """Return the coefficients for a single field at a given time level.

        Must be implemented in child classes."""
        raise NotImplementedError

    @abstractmethod
    def tesselate(self, field, level=0):
        """Return a tesselation (for plotting) of a single field at a given time level.

        Must be implemented in child classes."""
        raise NotImplementedError

    @abstractmethod
    def sink(self, *args, **kwargs):
        """Create a data sink (an output class) that matches the source.

        Must be implemented in child classes."""
        raise NotImplementedError

    def __str__(self):
        return '{}(ntimes={}, pardim={}, fields=[{}])'.format(
            self.__class__.__name__,
            self.ntimes,
            self.pardim,
            ','.join('{}({},{})'.format(f.name, f.size, f.ncomps)
                     for f in self.fields()),
        )


class DataSink:
    """There's no common interface for data sinks yet."""
    pass
