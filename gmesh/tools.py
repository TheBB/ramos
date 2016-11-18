from io import StringIO
from os.path import splitext
import importlib
import numpy as np
from matplotlib import pyplot
import sys
import splipy.IO
from itertools import chain, product, islice, repeat, tee
from multiprocessing import Pool
from . import data


class G2Object(splipy.IO.G2):

    def __init__(self, fstream):
        self.fstream = fstream
        super(G2Object, self).__init__('')

    def __enter__(self):
        self.onlywrite = False
        return self

def obj_to_string(obj):
    s = StringIO()
    with G2Object(s) as f:
        f.write(obj)
    return s.getvalue()


def structure(fn, out, coords, nums, level=0, store_basis=True):
    vtk = importlib.import_module('vtk')

    f = next(data.read(fn))
    dataset = f.reader.GetOutput()

    # Substitute unknown bounds with bounding box
    xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
    bbox = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    for i, (c, bb) in enumerate(zip(coords, bbox)):
        if c[0] is None: c[0] = bb[0]
        if c[1] is None: c[1] = bb[1]
        if c[0] == c[1]: nums[i] = 0

    dims = 0
    shape = []
    for i, (c, n) in enumerate(zip(coords, nums)):
        coords[i] = np.linspace(c[0], c[1], n+1)
        if n > 0:
            dims += 1
            shape.append(n+1)

    points = vtk.vtkPoints()
    for z, y, x in product(*coords[::-1]):
        points.InsertNextPoint(x, y, z)
    new_grid = vtk.vtkStructuredGrid()
    new_grid.SetDimensions(*[n+1 for n in nums])
    new_grid.SetPoints(points)

    probefilter = vtk.vtkProbeFilter()
    probefilter.SetSourceConnection(f.reader.GetOutputPort())
    probefilter.SetInputData(new_grid)
    probefilter.Update()
    structgrid = probefilter.GetStructuredGridOutput()

    basename, ext = splitext(out)

    if ext == '.hdf5':
        h5py = importlib.import_module('h5py')
        pointdata = structgrid.GetPointData()
        fields = {}
        for i in range(pointdata.GetNumberOfArrays()):
            fieldname = pointdata.GetArrayName(i)
            array = pointdata.GetArray(i)
            coefs = np.zeros((np.prod(shape), len(array.GetTuple(0))))
            for i, _ in enumerate(product(*coords[::-1])):
                coefs[i,:] = array.GetTuple(i)
            fields[fieldname] = coefs

        obj = {
            1: splipy.Curve,
            2: splipy.Surface,
            3: splipy.Volume,
        }[dims]()
        obj.set_dimension(3)
        obj.refine(*[k-2 for k in shape])
        for stuff in product(*map(enumerate, coords)):
            idx = tuple([i for (i,_),c in zip(stuff, coords) if len(c) > 1] + [None])
            obj.controlpoints[idx] = tuple(c for _,c in stuff)

        with h5py.File(basename + '.hdf5') as f:
            if store_basis:
                basis = f.require_group('/{}/basis/basis'.format(level))
                ints = np.fromstring(obj_to_string(obj), dtype=np.int8)
                basis.create_dataset('1', data=ints, dtype=np.int8)
            patch = f.require_group('/{}/1'.format(level))
            for fname, coefs in fields.items():
                if fname.startswith('vtk'):
                    continue
                patch.create_dataset(fname, data=coefs.flat)

        return fields

    elif ext == '.vtk':
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileName(out)
        writer.SetInputData(structgrid)
        writer.Write()

        return {}


def reduce(fields, filenames):
    objs = list(chain.from_iterable(data.read(fn) for fn in filenames))
    coeffs = []
    for obj in objs:
        for t in range(0, obj.ntimes):
            for f in fields:
                coeffs.append(obj.coefficients(f, t, 0))
    data_mx = np.hstack(coeffs)

    _, s, v = np.linalg.svd(data_mx.T, full_matrices=False)
    pyplot.semilogy(s)
    pyplot.show()
