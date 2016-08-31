import h5py
import utm
import numpy as np


ZONE = 33
NORTH = True
RES = 10


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
            print(nx, ny)
            x = np.linspace(self.west, self.east, (nx-1)//RES+1)
            y = np.linspace(self.north, self.south, (ny-1)//RES+1)
            print(x.shape, y.shape)
            xx, yy = np.meshgrid(x, y)
            print(xx.shape, yy.shape)
            self.lats, self.lons = utm.to_latlon(xx, yy, ZONE, northern=NORTH)
            print(self.lats.shape, self.lons.shape)
            self.data = self.h5f['maps'][self.group]['data'][::RES,::RES]
            print(self.data.shape)


def read(fn):
    h5f = h5py.File(fn, 'r+')
    for group in h5f['maps']:
        yield HDF5Submap(h5f, group)
    # h5f.close()
