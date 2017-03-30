from vtk import vtkDataSetReader
from vtk.util.numpy_support import vtk_to_numpy

from ramos.io.DataSource import DataSource
from ramos.utils.vtk import mass_matrix


class VTKFiles(DataSource):

    def __init__(self, files):
        self.files = files

        dataset = self.dataset(0)
        xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
        variates = [xmin != xmax, ymin != ymax, zmin != zmax]
        pardim = sum(variates)
        super(VTKFiles, self).__init__(pardim, len(files))
        self.variates = [i for i, v in enumerate(variates) if v]

        pointdata = dataset.GetPointData()
        for i in range(pointdata.GetNumberOfArrays()):
            name = pointdata.GetArrayName(i)
            ncomps = pointdata.GetAbstractArray(i).GetNumberOfComponents()
            size = pointdata.GetAbstractArray(i).GetNumberOfTuples()
            self.add_field(name, ncomps, size)

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
