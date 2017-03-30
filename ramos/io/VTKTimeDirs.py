from os import listdir
from os.path import join
from vtk import vtkDataSetReader
from vtk.util.numpy_support import vtk_to_numpy

from ramos.io.DataSource import DataSource
from ramos.utils.vtk import mass_matrix


class VTKTimeDirs(DataSource):

    def __init__(self, paths):
        self.paths = paths

        files = set(listdir(paths[0]))
        for path in paths[1:]:
            files = files & set(listdir(path))
        self.files = list(files)

        dataset = self.dataset(0, 0)
        xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
        variates = [xmin != xmax, ymin != ymax, zmin != zmax]
        pardim = sum(variates)
        super(VTKTimeDirs, self).__init__(pardim, len(paths))
        self.variates = [i for i, v in enumerate(variates) if v]

        for fi in range(len(self.files)):
            dataset = self.dataset(0, fi)
            pointdata = dataset.GetPointData()
            for i in range(pointdata.GetNumberOfArrays()):
                name = pointdata.GetArrayName(i)
                ncomps = pointdata.GetAbstractArray(i).GetNumberOfComponents()
                size = pointdata.GetAbstractArray(i).GetNumberOfTuples()
                self.add_field(name, ncomps, size, file_index=fi)

    def dataset(self, path_index, file_index):
        reader = vtkDataSetReader()
        reader.SetFileName(join(self.paths[path_index], self.files[file_index]))
        reader.Update()
        return reader.GetOutput()

    def field_mass_matrix(self, field):
        return mass_matrix(self.dataset(0, field.file_index), self.variates)

    def field_coefficients(self, field, level=0):
        dataset = self.dataset(level, field.file_index)
        pointdata = dataset.GetPointData()
        array = pointdata.GetAbstractArray(field.name)
        return vtk_to_numpy(array)
