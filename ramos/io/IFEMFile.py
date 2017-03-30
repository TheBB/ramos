import h5py
from io import StringIO
from itertools import chain, product
from lxml import etree
import numpy as np
from os.path import splitext
import splipy.IO

from ramos.io.DataSource import DataSource
from ramos.utils.splipy import mass_matrix


class G2Object(splipy.IO.G2):

    def __init__(self, fstream, mode):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self):
        return self


class IFEMFile(DataSource):

    def __init__(self, filename):
        self.hdf_filename = filename

        xml_filename = splitext(filename)[0] + '.xml'
        self.xml = etree.parse(xml_filename)

        with self.hdf5() as f:
            basis = next(iter(f['0/basis']))
            patch = self.patch(basis, 0)
            pardim = patch.pardim
            ntimes = len(f)

        assert pardim == 2
        super(IFEMFile, self).__init__(pardim, ntimes)

        for xmlf in self.xml.findall("./entry[@type='field']"):
            name = xmlf.attrib['name']
            ncomps = int(xmlf.attrib['components'])
            basis = xmlf.attrib['basis']
            size = sum(len(p) for p in self.patches(basis))
            self.add_field(name, ncomps, size, basis=basis)

    def hdf5(self):
        return h5py.File(self.hdf_filename, 'r')

    def npatches(self, basis):
        with self.hdf5() as f:
            return len(f['0/basis/{}'.format(basis)])

    def patch(self, basis, index):
        with self.hdf5() as f:
            g2str = f['0/basis/{}/{}'.format(basis, index+1)][:].tobytes().decode()
            g2data = StringIO(g2str)
            with G2Object(g2data, 'r') as g:
                return g.read()[0]

    def patches(self, basis):
        for i in range(self.npatches(basis)):
            yield self.patch(basis, i)

    def field_mass_matrix(self, field):
        glob_index = 0

        ret = []
        for patch in self.patches(field.basis):
            ret.append(mass_matrix(patch, glob_index))
            glob_index += len(patch)

        return tuple(
            np.array(list(chain.from_iterable(r[i] for r in ret)))
            for i in range(3)
        )

    def field_coefficients(self, field, level=0):
        npatches = self.npatches(field.basis)
        with self.hdf5() as f:
            return np.hstack([
                f['{}/{}/{}'.format(level, pid+1, field.name)][:]
                for pid in range(npatches)
            ])
