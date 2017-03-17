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
        ncomps = sum(d['ncomps'] for d in self.data)
        for d in self.data:
            data.extend([d['data']]*d['ncomps'])
            for _ in range(d['ncomps']):
                rows.append(ncomps * d['rows'] + len(rows))
                cols.append(ncomps * d['cols'] + len(cols))

        data, rows, cols = map(np.hstack, (data, rows, cols))
        mx = csr_matrix((data, (rows, cols)))
        mx.sum_duplicates()
        return mx
