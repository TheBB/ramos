from itertools import chain
import numpy as np
from scipy.sparse import csr_matrix


class MatrixBuilder:

    def __init__(self):
        self.data = []

    def add(self, data, rows, cols, ncomps=1, scale=1.0):
        self.data.append({
            'data': data * scale,
            'rows': rows,
            'cols': cols,
            'ncomps': ncomps,
        })

    def build(self):
        data, rows, cols = [], [], []
        glob_index = 0
        for d in self.data:
            ncomps = d['ncomps']
            data.extend([d['data']] * ncomps)
            for i in range(ncomps):
                rows.append(ncomps * d['rows'] + glob_index + i)
                cols.append(ncomps * d['cols'] + glob_index + i)
            glob_index += ncomps * len(d['data'])

        data, rows, cols = map(np.hstack, (data, rows, cols))
        mx = csr_matrix((data, (rows, cols)))
        mx.sum_duplicates()
        return mx
