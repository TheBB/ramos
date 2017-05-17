from os import makedirs
from os.path import exists, isdir, join, split
from vtk import vtkDataSetReader
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from ramos.io.Base import DataSource, DataSink
from ramos.utils.vectors import decompose
from ramos.utils.vtk import mass_matrix, write_to_file


class VTKFilesSource(DataSource):

    def __init__(self, files):
        self.files = files

        dataset = self.dataset(0)
        xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
        variates = [
            abs(a - b) > 1e-5
            for a, b in ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        ]
        pardim = sum(variates)
        super(VTKFilesSource, self).__init__(pardim, len(files))
        self.variates = [i for i, v in enumerate(variates) if v]

        pointdata = dataset.GetPointData()
        for i in range(pointdata.GetNumberOfArrays()):
            name = pointdata.GetArrayName(i)
            ncomps = pointdata.GetAbstractArray(i).GetNumberOfComponents()
            size = pointdata.GetAbstractArray(i).GetNumberOfTuples()
            self.add_field(name, ncomps, size)

    def datasets():
        return ((i, self.dataset(i)) for i in range(len(self.files)))

    def dataset(self, index):
        reader = vtkDataSetReader()
        reader.SetFileName(self.files[index])
        reader.Update()
        return reader.GetOutput()

    def field_mass_matrix(self, field):
        return mass_matrix(self.dataset(0), self.variates)

    def field_coefficients(self, field, level=0):
        dataset = self.dataset(level)
        pointdata = dataset.GetPointData()
        array = pointdata.GetAbstractArray(field.name)
        return vtk_to_numpy(array)

    def tesselate(self, field, level=0):
        field = self.field(field)
        dataset = self.dataset(level)
        points = vtk_to_numpy(dataset.GetPoints().GetData())
        x, y = (points[...,i] for i in self.variates)
        coeffs = vtk_to_numpy(dataset.GetPointData().GetAbstractArray(field.name))

        cell_indices = get_cell_indices(dataset)
        if cell_indices.shape[1] > 3:
            return (x, y), coeffs
        return (x, y, cell_indices), coeffs

    def sink(self, *args, **kwargs):
        return VTKFilesSink(self, *args, **kwargs)


class VTKFilesSink(DataSink):

    def __init__(self, parent, path, basename='mode'):
        self.parent = parent
        self.path = path
        self.basename = basename
        self.files = []

    def __enter__(self):
        if not exists(self.path):
            makedirs(self.path)
        if not isdir(self.path):
            raise IOError
        return self

    def __exit__(self, type_, value, backtrace):
        pass

    def add_level(self, time):
        self.files.append('{}-{}.vtk'.format(join(self.path, self.basename), len(self.files)))

    def write_fields(self, level, coeffs, fields):
        fields = [self.parent.field(f) for f in fields]
        field_coeffs = decompose(fields, coeffs)

        dataset = self.parent.dataset(level)
        pointdata = dataset.GetPointData()
        while pointdata.GetNumberOfArrays() > 0:
            pointdata.RemoveArray(0)
        for field, coeffs in zip(fields, field_coeffs):
            array = numpy_to_vtk(coeffs, deep=1)
            array.SetName(field.name)
            pointdata.AddArray(array)

        write_to_file(dataset, self.files[level])
