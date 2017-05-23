"""Collection of commonly used parallel worker functions."""

import numpy as np


def energy_content(source, level, field, mx):
    """Compute the energy content of a field.

    - `source`: the data source
    - `level`: the time level to read from
    - `field`: the name of the field to read
    - `mx`: mass matrix

    Equivalent of u^T × M × u, where M is the mass matrix and U is the
    coefficient vector.
    """
    coeffs = source.coefficients([field], level)
    return mx.dot(coeffs).dot(coeffs)


def normalized_coeffs(source, level, fields, mass):
    """Returns the coefficient vector of a number of fields, centered around
    the component-wise mean.

    - `source`: the data source
    - `level`: the time level to read from
    - `fields`: the name of the fields to read
    - `mx`: dict mapping field names to mass matrices
    """
    ret = []
    for field in fields:
        mx = mass[field]
        coeffs = source.coefficients(field, level, flatten=False)
        mean = np.sum(mx.dot(coeffs), axis=0) / mx.sum()  # Compute component-wise means,
        coeffs -= mean                                    # and subtract them.
        ret.append(np.ndarray.flatten(coeffs))            # The result needs to be flattened.
    return np.hstack(coeffs)


def mv_dot(vec, mx):
    """Computes a matrix-vector product."""
    return mx.dot(vec)


def vv_dot(va, vb):
    """Computes a vector-vector dot product."""
    return va.dot(vb)
