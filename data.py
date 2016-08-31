import h5py
import utm
import numpy as np


ZONE = 33
NORTH = True


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


def read(fn):
    h5f = h5py.File(fn, 'r+')
    for group in h5f['maps']:
        yield HDF5Submap(h5f, group)
    # h5f.close()
