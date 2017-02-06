from collections import namedtuple
from io import StringIO
import xml.etree.ElementTree as xml
import importlib
from . import utm
import splipy.IO
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

    class G2Object(splipy.IO.G2):
        def __init__(self, fstream, mode):
            self.fstream = fstream
            self.onlywrite = mode == 'w'
            super(IFEMFile.G2Object, self).__init__('')
        def __enter__(self):
            return self

    @staticmethod
    def obj_to_string(obj):
        s = StringIO()
        with IFEMFile.G2Object(s, 'w') as f:
            f.write(obj)
        return s.getvalue()

    def __init__(self, fn):
        h5py = importlib.import_module('h5py')
        self.h5f = h5py.File(fn)
        self._bases = {}

        self.xml_fn = splitext(fn)[0] + '.xml'
        try:
            self.dom = xml.parse(self.xml_fn).getroot()
        except FileNotFoundError:
            self.dom = xml.Element('info')

    @property
    def fields(self):
        yield from self.dom.findall("./entry[@type='field']")

    def write_xml(self):
        xml.ElementTree(self.dom).write(self.xml_fn, encoding='utf-8', xml_declaration=True)

    def basis(self, name, patchid):
        try:
            return self._bases[name, patchid]
        except KeyError:
            g2str = self.h5f['0/basis/{}/{}'.format(name, patchid+1)][:].tobytes().decode()
            g2data = StringIO(g2str )
            with IFEMFile.G2Object(g2data, 'r') as g:
                obj = g.read()[0]
                self._bases[name, patchid] = obj
                return obj

    def field(self, name):
        return self.dom.findall("./entry[@type='field'][@name='{}']".format(name))[0]

    def set_meta(self, fieldname, name, value):
        field = self.field(fieldname)
        field.attrib['meta_' + name] = str(value)

    def get_meta(self, fieldname, name):
        field = self.field(fieldname)
        return field.attrib['meta_' + name]

    @property
    def ntimes(self):
        return len(self.h5f)

    @property
    def dt(self):
        return float(self.dom.findall('./timestep')[0].text)

    @property
    def t_start(self):
        return float(self.dom.findall('./timestep')[0].attrib['end'])

    @property
    def t_end(self):
        return float(self.dom.findall('./timestep')[0].attrib['start'])

    def t_at(self, i):
        i /= (self.ntimes - 1)
        return self.t_end * (1 - i) + self.t_start * i

    def save_basis(self, name, patchid, obj):
        self._bases[name, patchid] = obj
        group = self.h5f.require_group('/0/basis/{}'.format(name))
        ints = np.fromstring(IFEMFile.obj_to_string(obj), dtype=np.int8)
        pid = str(patchid + 1)
        if pid in group:
            del group[pid]
        group.create_dataset(pid, data=ints, dtype=np.int8)

    def coeffs(self, name, timelevel, patchid, vectorize=False):
        field = self.field(name).attrib
        components = int(field['components'])
        basis = self.basis(field['basis'], patchid)

        # Extract data in correct order (column-major)
        data = self.h5f[str(timelevel)][str(patchid + 1)][name][:]
        shape = tuple(list(basis.shape)[::-1] + [components])
        data = np.reshape(data, shape)

        # Transpose so we get it row-major
        axes = list(range(len(data.shape)))
        axes = axes[-2::-1] + [axes[-1]]
        data = np.transpose(data, axes)

        if vectorize:
            shape = (np.prod(data.shape[:-1]), components)
            return np.reshape(data, shape)
        return data

    def save_coeffs(self, name, basis, level, patchid, coeffs, transpose=False):
        if transpose:
            axes = list(range(len(coeffs.shape)))
            axes = axes[-2::-1] + [axes[-1]]
            coeffs = np.transpose(coeffs, axes)

        group = self.h5f.require_group('/{}/{}'.format(level, patchid+1))
        group.create_dataset(name, data=coeffs.flat)

        # Number of levels
        try:
            levels = self.dom.findall('./levels')[0]
            levels.text = str(max(int(levels.text), level+1))
        except IndexError:
            levels = xml.SubElement(self.dom, 'levels')
            levels.text = str(level + 1)

        # Field
        try:
            entry = self.dom.findall("./entry[@type='field'][@name='{}']".format(name))[0]
        except IndexError:
            entry = xml.SubElement(self.dom, 'entry')
        entry.attrib.update({
            'type': 'field',
            'name': name,
            'basis': basis,
            'components': str(coeffs.shape[-1]),
        })

        self.write_xml()

    def set_timestep(self, ts, start, end):
        try:
            timestep = self.dom.findall('./timestep')[0]
        except IndexError:
            timestep = xml.SubElement(self.dom, 'timestep')
        timestep.text = str(ts)
        timestep.attrib.update({
            'constant': '1',
            'order': '1',
            'interval': '1',
            'start': str(start),
            'end': str(end),
        })

        self.write_xml()


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
    elif ext in {'.h5', '.hdf5'}:
        yield IFEMFile(fn)
    elif ext == '.dem':
        raise NotImplementedError('DEM support not implemented')
    elif ext == '.nc':
        yield NetCDFFile(fn)
    elif ext == '.vtk':
        yield VTKFile(fn)
