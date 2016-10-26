from collections import namedtuple
from io import StringIO
import xml.etree.ElementTree as xml
import importlib
from . import utm
import numpy as np
import itertools
from os.path import splitext
from functools import reduce


ZONE = 33
NORTH = True
RES = 300
BD_RES = 15


def interpolate(array, *idxs):
    left = [np.floor(i).astype(int) for i in idxs]
    for i, l in enumerate(left):
        l[np.nonzero(l == array.shape[i] - 1)] -= 1
    ret = np.zeros(tuple(len(i) for i in idxs))
    for corner in itertools.product((False, True), repeat=len(idxs)):
        coef = np.ix_(*(j-l if c else 1-(j-l)
                        for l,j,c in zip(left, idxs, corner)))
        coef = reduce(np.multiply, coef)
        cidxs = tuple(l + 1 if c else l for l, c in zip(left, corner))
        ret += coef * array[np.meshgrid(*cidxs)].T
    return ret


class IFEMFile:

    Field = namedtuple('ResField', ['components', 'basis'])

    def __init__(self, fn):
        h5py = importlib.import_module('h5py')
        self.h5f = h5py.File(fn, 'r+')

        dom = xml.parse(splitext(fn)[0] + '.xml')
        self.fields = {}
        for child in dom.getroot():
            if child.tag == 'entry' and child.attrib['type'] == 'field':
                self.fields[child.attrib['name']] = IFEMFile.Field(
                    components=int(child.attrib['components']),
                    basis=child.attrib['basis']
                )

        io = importlib.import_module('splipy.io')
        class G2Object(io.G2):
            def __init__(self, fstream):
                self.fstream = fstream
                super(G2Object, self).__init__('')
            def __enter__(self):
                self.onlywrite = False
                return self

        self.bases = {}
        for basisname, data in self.h5f['0']['basis'].items():
            for patchid in range(0, len(data)):
                g2str = data[str(patchid + 1)][:].tobytes().decode()
                g2data = StringIO(data[str(patchid + 1)][:].tobytes().decode())
                with G2Object(g2data) as g:
                    self.bases.setdefault(basisname, [None]*len(data))[patchid] = g.read()[0]

    def coefficients(self, fieldname, patchid):
        field = self.fields[fieldname]
        shape = (np.prod(self.bases[field.basis][patchid].shape), field.components)
        data = self.h5f['0'][str(patchid + 1)][fieldname][:]
        return np.reshape(data, shape)


class HDF5Submap:

    def __init__(self, h5f, group):
        self.group = group
        self.h5f = h5f

        for k in ['west', 'east', 'south', 'north']:
            setattr(self, k, h5f['maps'][group][k][()])

        self.pts = [utm.to_latlon(self.west, self.south, ZONE, northern=NORTH)[::-1],
                    utm.to_latlon(self.east, self.south, ZONE, northern=NORTH)[::-1],
                    utm.to_latlon(self.east, self.north, ZONE, northern=NORTH)[::-1],
                    utm.to_latlon(self.west, self.north, ZONE, northern=NORTH)[::-1]]

    def contains(self, lat, lon):
        east, north, _, _ = utm.from_latlon(lat, lon, force_zone_number=ZONE)
        return self.west <= east <= self.east and self.south <= north <= self.north

    def compute(self):
        if not hasattr(self, 'data'):
            ny, nx = self.h5f['maps'][self.group]['data'].shape
            x = np.linspace(self.west, self.east, RES)
            y = np.linspace(self.north, self.south, RES)
            xx, yy = np.meshgrid(x, y)
            self.lats, self.lons = utm.to_latlon(xx, yy, ZONE, northern=NORTH)
            x = np.linspace(0, nx-1, RES)
            y = np.linspace(0, ny-1, RES)
            self.data = interpolate(self.h5f['maps'][self.group]['data'][()], y, x)


class NetCDFFile:

    def __init__(self, fn):
        nc4 = importlib.import_module('netCDF4')
        data = nc4.Dataset(fn, 'r')

        nx, ny = data['longitude'].shape
        x = np.linspace(0, nx-1, BD_RES)
        y = np.linspace(0, ny-1, BD_RES)
        lon = interpolate(data['longitude'][:], x, y)
        lat = interpolate(data['latitude'][:], x, y)
        lon = list(lon[:,0]) + list(lon[-1,:]) + list(lon[-1::-1,-1]) + list(lon[0,-1::-1])
        lat = list(lat[:,0]) + list(lat[-1,:]) + list(lat[-1::-1,-1]) + list(lat[0,-1::-1])

        path = importlib.import_module('matplotlib.path')
        self.path = path.Path(np.vstack((lat, lon)).T)
        self.pts = list(zip(lon, lat))

    def contains(self, lat, lon):
        return self.path.contains_point((lat, lon))

    def compute(self):
        raise NotImplementedError('Compute for NetCDF not implemented')


class VTKFile:

    def __init__(self, fn):
        vtk = importlib.import_module('vtk')
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(fn)
        reader.Update()
        self.reader = reader

    def contains(self):
        raise NotImplementedError('Contains for VTK not implemented')

    def compute(self):
        raise NotImplementedError('Compute for VTK not implemented')


def read(fn):
    ext = splitext(fn)[-1].lower()
    if ext == '.hms':
        h5py = importlib.import_module('h5py')
        h5f = h5py.File(fn, 'r+')
        for group in h5f['maps']:
            yield HDF5Submap(h5f, group)
    elif ext == '.hdf5':
        yield IFEMFile(fn)
    elif ext == '.dem':
        raise NotImplementedError('DEM support not implemented')
    elif ext == '.nc':
        yield NetCDFFile(fn)
    elif ext == '.vtk':
        yield VTKFile(fn)
