import h5py
from io import StringIO
from itertools import chain, product, repeat
from lxml import etree
import numpy as np
from os.path import splitext
import splipy.IO

from ramos.io.Base import DataSource, DataSink
from ramos.utils.splipy import mass_matrix
from ramos.utils.vectors import decompose


class G2Object(splipy.IO.G2):

    def __init__(self, fstream, mode):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self):
        return self


def obj_to_string(obj):
    s = StringIO()
    with IFEMFile.G2Object(s, 'w') as f:
        f.write(obj)
    return s.getvalue()


class IFEMFileSource(DataSource):

    def __init__(self, filename):
        self.hdf_filename = filename

        xml_filename = splitext(filename)[0] + '.xml'
        self.xml = etree.parse(xml_filename)

        bases = []
        with self.hdf5() as f:
            basis = next(iter(f['0/basis']))
            bases.append(basis)
            patch = self.patch(basis, 0)
            pardim = patch.pardim
            ntimes = len(f)

        variates = repeat(False)
        for basis in bases:
            for patch in self.patches(basis):
                cps = patch.controlpoints
                variates = [
                    v or abs(np.max(cps[...,d]) - np.min(cps[...,d])) > 1e-5
                    for d, v in zip(range(patch.dimension), variates)
                ]

        assert pardim == 2
        super(IFEMFileSource, self).__init__(pardim, ntimes)
        self.variates = [i for i, v in enumerate(variates) if v]

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

    def tesselate(self, field, level=0):
        field = self.field(field)
        coeffs = self.field_coefficients(field, level).reshape((field.size, field.ncomps))

        xs, ys, zs = [], [], []
        glob_index = 0
        for i, patch in enumerate(self.patches(field.basis)):
            params = patch.knots()
            pts = patch(*params)
            xs.append(np.ndarray.flatten(pts[..., self.variates[0]]))
            ys.append(np.ndarray.flatten(pts[..., self.variates[1]]))

            n = len(patch)
            cps = coeffs[glob_index:glob_index+n,:]
            shape = list(patch.shape)[::-1] + [field.ncomps]
            cps = np.reshape(cps, shape)

            axes = list(range(len(cps.shape)))
            axes = axes[-2::-1] + [axes[-1]]
            cps = np.transpose(cps, axes)
            patch.controlpoints = cps

            zs.append(np.reshape(patch(*params), (len(patch), field.ncomps)))

        return (np.hstack(xs), np.hstack(ys), np.vstack(zs))

    def sink(self, *args, **kwargs):
        return IFEMFileSink(self, *args, **kwargs)


class IFEMFileSink(DataSink):

    def __init__(self, parent, path):
        self.parent = parent
        self.hdf5_filename = path
        basename, _ = splitext(path)
        self.xml_filename = '{}.xml'.format(path)

    def __enter__(self):
        self.hdf5 = h5py.File(self.hdf5_filename, 'w')
        self.dom = etree.Element('info')

    def __exit__(self, type_, value, backtrace):
        self.hdf5.close()
        with open(self.xml_filename, 'wb') as f:
            f.write(etree.tostring(
                self.dom, pretty_print=True, encoding='utf-8',
                xml_declaration=True, standalone=True,
            ))

    def add_level(self, time):
        index = len(self.hdf5)
        self.hdf5.require_group(str(index))

    def ensure_basis(self, basis):
        grp = self.hdf5.require_group('0/basis')
        if basis in grp:
            return
        grp = grp.require_group(basis)
        for i, patch in enumerate(self.parent.patches(basis)):
            ints = np.fromstring(obj_to_string(patch), dtype=np.int8)
            pid = str(i + 1)
            if pid in grp:
                del grp[pid]
            grp.create_dataset(pid, data=ints, dtype=np.int8)

    def write_fields(self, level, coeffs, fields):
        fields = [self.parent.field(f) for f in fields]
        field_coeffs = decompose(fields, coeffs)

        for field, coeffs in zip(fields, field_coeffs):
            self.ensure_basis(field.basis)
            glob_index = 0
            for i, patch in enumerate(self.parent.patches(field.basis)):
                grp = self.hdf5.require_group('{}/{}'.format(level, i+1))
                if field.name in grp:
                    del grp[field.name]
                n = len(patch) * field.ncomps
                grp.create_dataset(field.name, data=coeffs[glob_index:glob_index+n])
                glob_index += n
