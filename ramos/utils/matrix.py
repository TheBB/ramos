from itertools import chain
import numpy as np
from scipy.sparse import csr_matrix


class MatrixBuilder:
    """Utility class for constructing mass matrices.

    To use, call the add() method as many times as necessary, then the build()
    method to construct a Scipy sparse CSR format matrix.
    """

    def __init__(self):
        self.data = []

    def add(self, data, rows, cols, ncomps=1, scale=1.0):
        """Add data to this matrix.

        `data` is a one-dimensional numpy array of matrix elements.
        `rows` is a one-dimensional numpy array of row indices.
        `cols` is a one-dimensional numpy array of column indices.
        `ncomps` is an optional number of components (multiple components has
        the effect of "duplicating" the data).
        `scale` is an optional scaling factor to apply.
        """
        self.data.append({
            'data': data * scale,
            'rows': rows,
            'cols': cols,
            'ncomps': ncomps,
        })

    def build(self):
        """Finalize the matrix."""

        # First, we need to create global data, rows and cols arrays.
        data, rows, cols = [], [], []
        glob_index = 0
        for d in self.data:
            ncomps = d['ncomps']
            data.extend([d['data']] * ncomps)

            # The rows and cols arrays are locally indexed from zero, so we
            # must transform them to the global matrix index system.
            for i in range(ncomps):
                rows.append(ncomps * d['rows'] + glob_index + i)
                cols.append(ncomps * d['cols'] + glob_index + i)
            glob_index += ncomps * len(d['data'])

        data, rows, cols = map(np.hstack, (data, rows, cols))
        mx = csr_matrix((data, (rows, cols)))
        mx.sum_duplicates()
        return mx
