from itertools import product, repeat, chain
import logging
from multiprocessing import Pool
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from ramos.utils.quadrature import triangular


def get_cells(dataset):
    if isinstance(dataset, vtk.vtkUnstructuredGrid):
        return dataset.GetCells()
    elif isinstance(dataset, vtk.vtkPolyData):
        return dataset.GetPolys()
    else:
        raise TypeError('Unknown dataset type')


def get_cell_indices(dataset):
    # cell_indices: each row contains the point indices for that cell,
    # possibly filled with -1 on the end for variable cell sizes
    cells = get_cells(dataset)
    ncells = cells.GetNumberOfCells()
    cellsize = cells.GetMaxCellSize()
    cell_indices = np.empty((ncells, cellsize), dtype=np.int)
    cell_indices[:] = -1
    cell_raw = vtk_to_numpy(cells.GetData())
    j = 0
    for i in range(ncells):
        l = cell_raw[j]
        j += 1
        cell_indices[i,:l] = cell_raw[j:j+l]
        j += l
    return cell_indices


def decompose(dataset, variates):
    cell_indices = get_cell_indices(dataset)
    ncells, cellsize = cell_indices.shape
    logging.debug('Mesh with %d cells, max size %d', ncells, cellsize)

    # cell_points: each row contains a npts × 3 matrix with actual points,
    # possibly filled with NaNs on the end for variable cell sizes
    points = dataset.GetPoints()
    point_raw = vtk_to_numpy(points.GetData())

    rows, cols = np.where(cell_indices >= 0)
    cell_points = np.empty((ncells, cellsize, 3))
    cell_points[:] = np.NAN
    cell_points[rows, cols, :] = point_raw[cell_indices[rows, cols]]
    cell_points = cell_points[:, :, variates]

    return cell_indices, cell_points


def element_mass_matrix(indices, points, pardim):
    npts = sum(indices >= 0)

    # Each branch defines quadrature, jac and basis
    if npts == 4 and pardim == 2:
        qpts, qwts = np.polynomial.legendre.leggauss(3)
        qpts = (qpts + 1) / 2
        qwts /= 2
        def quadrature():
            for quad in product(zip(qpts, qwts), repeat=pardim):
                pt = tuple(q[0] for q in quad)
                wt = np.prod([q[1] for q in quad])
                yield pt, wt
        quadrature = quadrature()
        def jac(x, y):
            return np.linalg.det(np.array([[y-1, -y, y, 1-y], [x-1, 1-x, x, -x]]).dot(points))
        basis = [
            lambda x, y: (1-x)*(1-y),
            lambda x, y: (1-x)*y,
            lambda x, y: x*y,
            lambda x, y: x*(1-y),
        ]

    elif npts == 3 and pardim == 2:
        qpts, qwts = triangular(2)
        quadrature = zip(qpts, qwts)
        cjac = np.linalg.det(np.vstack([points[2,:] - points[0,:], points[1,:] - points[0,:]]))
        jac = lambda x, y: cjac
        basis = [
            lambda x, y: 1-x-y,
            lambda x, y: y,
            lambda x, y: x,
        ]

    else:
        return [], [], []

    # Actual quadrature loop
    result = [0.0 for _ in range(npts*npts)]
    for pt, wt in quadrature:
        for i, bfs in enumerate(product(basis, repeat=2)):
            j = abs(jac(*pt))
            result[i] += wt * bfs[0](*pt) * bfs[1](*pt) * j

    indices = list(product(indices[np.where(indices >= 0)], repeat=2))
    row_inds = [i[0] for i in indices]
    col_inds = [i[1] for i in indices]

    return result, row_inds, col_inds


def mass_matrix(dataset, variates, parallel=True):
    # Decompose the grid into arrays of indices and points for each cell
    indices, points = decompose(dataset, variates)

    # Each worker returns a tuple of: flat matrix, row indices, column indices
    args = zip(indices, points, repeat(len(variates)))
    if parallel:
        pool = Pool()
        ret = pool.starmap(element_mass_matrix, args)
    else:
        ret = [element_mass_matrix(*arg) for arg in args]

    # Form complete data arrays
    return tuple(
        np.array(list(chain.from_iterable(r[i] for r in ret)))
        for i in range(3)
    )


def write_to_file(dataset, filename):
    if isinstance(dataset, vtk.vtkPolyData):
        writer = vtk.vtkPolyDataWriter()
    elif isinstance(dataset, vtk.vtkUnstructuredGrid):
        writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(dataset)
    writer.Write()
