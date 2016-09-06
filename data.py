import netCDF4 as nc4
import h5py
import utm
import numpy as np
import itertools
from matplotlib.path import Path
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
        data = nc4.Dataset(fn, 'r')

        nx, ny = data['longitude'].shape
        x = np.linspace(0, nx-1, BD_RES)
        y = np.linspace(0, ny-1, BD_RES)
        lon = interpolate(data['longitude'][:], x, y)
        lat = interpolate(data['latitude'][:], x, y)
        lon = list(lon[:,0]) + list(lon[-1,:]) + list(lon[-1::-1,-1]) + list(lon[0,-1::-1])
        lat = list(lat[:,0]) + list(lat[-1,:]) + list(lat[-1::-1,-1]) + list(lat[0,-1::-1])

        self.path = Path(np.vstack((lat, lon)).T)
        self.pts = list(zip(lon, lat))

    def contains(self, lat, lon):
        return self.path.contains_point((lat, lon))

    def compute(self):
        pass

def read(fn):
    ext = splitext(fn)[-1].lower()
    if ext == '.hms':
        h5f = h5py.File(fn, 'r+')
        for group in h5f['maps']:
            yield HDF5Submap(h5f, group)
    elif ext == '.dem':
        print('DEM support not added yet')
    elif ext == '.nc':
        yield NetCDFFile(fn)
