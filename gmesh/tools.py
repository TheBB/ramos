from io import StringIO
from os.path import splitext
import importlib
import numpy as np
import sys
import splipy.io
from itertools import product, islice, repeat, tee
from multiprocessing import Pool
from . import data


class G2Object(splipy.io.G2):

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


def structure(fn, out, nx, ny, nz, xval, yval, zval,):
    vtk = importlib.import_module('vtk')

    f = next(data.read(fn))
    dataset = f.reader.GetOutput()

    xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()

    dims = 0
    shape = []
    bounds = []
    if isinstance(xval, float):
        xs = [xval]
    else:
        xs = np.linspace(xmin, xmax, nx+1)
        dims += 1
        shape.append(nx+1)
        bounds.append((xmin, xmax))
    if isinstance(yval, float):
        ys = [xval]
    else:
        ys = np.linspace(ymin, ymax, ny+1)
        dims += 1
        shape.append(ny+1)
        bounds.append((ymin, ymax))
    if isinstance(yval, float):
        zs = [xval]
    else:
        zs = np.linspace(zmin, zmax, nz+1)
        dims += 1
        shape.append(nz+1)
        bounds.append((zmin, zmax))

    points = vtk.vtkPoints()
    for z, y, x in product(zs, ys, xs):
        points.InsertNextPoint(x, y, z)
    new_grid = vtk.vtkStructuredGrid()
    new_grid.SetDimensions(nx+1, ny+1, nz+1)
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
            coefs = np.zeros((len(zs) * len(ys) * len(xs), len(array.GetTuple(0))))
            for i, _ in enumerate(product(zs, ys, xs)):
                coefs[i,:] = array.GetTuple(i)
            fields[fieldname] = coefs
        with open(basename + '.xml', 'w') as f:
            f.write('<stuff>\n')
            f.write('  <levels>1</levels>\n')
            for fname, coefs in fields.items():
                f.write('  <entry type="field" name="{}" basis="basis" components="{}" />\n'.format(
                    fname, coefs.shape[-1]
                ))
            f.write('</stuff>\n')

        if dims == 1:
            obj = splipy.Curve()
        elif dims == 2:
            obj = splipy.Surface()
        elif dims == 3:
            obj = splipy.Volume()
        obj.refine(*[k-2 for k in shape])
        obj.scale(*[b-a for a,b in bounds])

        with h5py.File(basename + '.hdf5', 'w') as f:
            basis = f.create_group('/0/basis/basis')
            ints = np.fromstring(obj_to_string(obj).encode('utf-8'), dtype=np.int8)
            basis.create_dataset('1', data=ints, dtype='i8')
            level = f.create_group('/0/1')
            for fname, coefs in fields.items():
                level.create_dataset(fname, data=coefs.flat)

    elif ext == '.vtk':
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileName(out)
        writer.SetInputData(structgrid)
        writer.Write()
