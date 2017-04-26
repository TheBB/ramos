import numpy as np


def energy_content(source, level, field, mx):
    coeffs = source.coefficients([field], level)
    return mx.dot(coeffs).dot(coeffs)


def normalized_coeffs(source, level, fields, mass):
    ret = []
    for field in fields:
        mx = mass[field]
        coeffs = source.coefficients(field, level, flatten=False)
        mean = np.sum(mx.dot(coeffs), axis=0) / mx.sum()
        coeffs -= mean
        ret.append(np.ndarray.flatten(coeffs))
    return np.hstack(coeffs)


def mv_dot(vec, mx):
    return mx.dot(vec)


def vv_dot(va, vb):
    return va.dot(vb)
