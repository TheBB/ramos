from itertools import groupby
import numpy as np
from os import listdir, makedirs
from os.path import exists, isdir, join
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from ramos.io.Base import DataSource, DataSink
from ramos.utils.vtk import mass_matrix


class VTKTimeDirsSource(DataSource):

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
        super(VTKTimeDirsSource, self).__init__(pardim, len(paths))
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
        reader = vtk.vtkDataSetReader()
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

    def sink(self, *args, **kwargs):
        return VTKTimeDirsSink(self, *args, **kwargs)


class VTKTimeDirsSink(DataSink):

    def __init__(self, parent, path):
        self.parent = parent
        self.path = path
        self.paths = []

    def __enter__(self):
        if not exists(self.path):
            makedirs(self.path)
        if not isdir(self.path):
            raise IOError
        return self

    def __exit__(self, type, value, backtrace):
        pass

    def add_level(self, time):
        path = join(self.path, str(time))
        if not exists(path):
            makedirs(path)
        if not isdir(self.path):
            raise IOError
        self.paths.append(path)

    def write_fields(self, level, coeffs, fields):
        data = []
        glob_index = 0
        for field in fields:
            field = self.parent.field(field)
            n = field.size * field.ncomps
            c = np.reshape(coeffs[glob_index:glob_index+n], (field.size, field.ncomps))
            data.append((field, c))
            glob_index += n

        key = lambda d: d[0].file_index
        data = sorted(data, key=key)
        for file_index, field_data in groupby(data, key):
            dataset = self.parent.dataset(0, file_index)
            pointdata = dataset.GetPointData()
            while pointdata.GetNumberOfArrays() > 0:
                pointdata.RemoveArray(0)
            for field, coeffs in field_data:
                array = numpy_to_vtk(coeffs, deep=1)
                array.SetName(field.name)
                pointdata.AddArray(array)

            if isinstance(dataset, vtk.vtkPolyData):
                writer = vtk.vtkPolyDataWriter()
            elif isinstance(dataset, vtk.vtkUnstructuredGrid):
                writer = vtk.vtkUnstructuredGridWriter()
            writer.SetFileName(join(self.paths[level], self.parent.files[file_index]))
            writer.SetInputData(dataset)
            writer.Write()
