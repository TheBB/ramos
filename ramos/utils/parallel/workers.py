import numpy as np


def energy_content(source, level, field, mx):
    coeffs = source.coefficients([field], level)
    return mx.dot(coeffs).dot(coeffs)


def normalized_coeffs(source, level, fields, mx, area):
    coeffs = source.coefficients(fields, level)
    scale = np.sum(mx.dot(coeffs)) / area
    coeffs -= scale
    return coeffs


def correlation(ca, cb, mx):
    return mx.dot(ca).dot(cb)
