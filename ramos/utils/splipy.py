from functools import reduce
from itertools import chain, product
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

from ramos.utils.parallel import parmap


def element_mass_matrix(element, quadrature, patch, glob_index):
    qpts = [(x+1)/2 * (R-L) + L for (x, _), (R, L) in zip(quadrature, element)]
    qwts = [w/2 * (R-L) for (_, w), (R, L) in zip(quadrature, element)]
    W = reduce(np.multiply.outer, qwts)

    # Assume pardim == 2
    du = patch.derivative(*qpts, d=(1,0))
    dv = patch.derivative(*qpts, d=(0,1))
    J = np.cross(du, dv)
    if patch.dimension == 3:
        J = np.linalg.norm(J, axis=2)
    else:
        J = np.abs(J)
    mass = J * W

    loc_indices = []
    for i, (basis, pts) in enumerate(zip(patch.bases, qpts)):
        N = basis.evaluate(pts, sparse=True).tocoo()
        loc_indices.append(min(N.col))
        N = coo_matrix((N.data, (N.row, N.col - min(N.col)))).todense()
        N = np.einsum('ki,kj->ijk', N, N)
        mass = np.tensordot(mass, N, ((0,), (2,)))

    axes = list(range(0, 2*patch.pardim, 2)) + list(range(1, 2*patch.pardim, 2))
    mass = np.transpose(mass, axes=axes)

    # Forgive me
    bfun_indices = [
        np.array(k) + i + glob_index
        for i, k in zip(loc_indices, zip(*product(*(range(o) for o in patch.order()))))
    ]
    indices = np.ravel_multi_index(bfun_indices, patch.shape, order='F')
    indices = list(product(indices, repeat=2))
    row_inds = [i[0] for i in indices]
    col_inds = [i[1] for i in indices]
    return np.ndarray.flatten(mass), row_inds, col_inds


def mass_matrix(patch, glob_index, parallel=True):
    spans = [list(zip(k[:-1], k[1:])) for k in patch.knots()]
    quadrature = [np.polynomial.legendre.leggauss(order + 1) for order in patch.order()]

    constant = (quadrature, patch, glob_index)
    if parallel:
        ret = parmap(element_mass_matrix, list(product(*spans)), constant, unwrap=False)
    else:
        ret = [element_mass_matrix(span, *constant) for span in product(*spans)]

    return tuple(
        np.array(list(chain.from_iterable(r[i] for r in ret)))
        for i in range(3)
    )
