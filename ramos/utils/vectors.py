import numpy as np


def decompose(fields, coeffs):
    ret = []
    glob_index = 0
    for field in fields:
        n = field.size * field.ncomps
        c = np.reshape(coeffs[glob_index:glob_index+n], (field.size, field.ncomps))
        ret.append(c)
        glob_index += n
    return ret
