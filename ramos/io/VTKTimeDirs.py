from itertools import groupby
import numpy as np
from os import listdir, makedirs
from os.path import exists, isdir, join
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from ramos.io.Base import DataSource, DataSink
from ramos.utils.mesh import mesh_filter
from ramos.utils.vectors import decompose
from ramos.utils.vtk import mass_matrix, write_to_file, get_cell_indices


class VTKTimeDirsSource(DataSource):

    def __init__(self, paths):
        """VTKTimeDirsSource reads this type of structure:

        <time1>/<file1>.vtk
        ...
        <time1>/<filen>.vtk
        <time2>/<file1>.vtk
        ...
        <time2>/<filen>.vtk
        ...
        <timen>/...

        The exact same filenames must be present in every time directory, and
        the directory names must be valid floating point numbers.

        `paths` is a list of pathnames to consider.
        """
        self.paths = paths

        # Find filenames that are present in all the directories.
        files = set(listdir(paths[0]))
        for path in paths[1:]:
            files = files & set(listdir(path))
        self.files = list(files)

        # Try to figure out how many parametric dimensions this data has.
        # Do this by loading any dataset and inspecting its bounding box.
        # (Not foolproof.)
        dataset = self.dataset(0, 0)
        xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
        variates = [
            abs(a - b) > 1e-5
            for a, b in ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        ]
        pardim = sum(variates)

        # Call superclass constructor
        super(VTKTimeDirsSource, self).__init__(pardim, len(paths))

        # List of physical dimensions that are also parametric dimensions
        self.variates = [i for i, v in enumerate(variates) if v]

        # Discover all the fields, by inspecting the files in the first time
        # level. No checking is done to make sure this info is consistent with
        # other time levels.
        for fi in range(len(self.files)):
            dataset = self.dataset(0, fi)
            pointdata = dataset.GetPointData()
            for i in range(pointdata.GetNumberOfArrays()):
                name = pointdata.GetArrayName(i)
                ncomps = pointdata.GetAbstractArray(i).GetNumberOfComponents()
                size = pointdata.GetAbstractArray(i).GetNumberOfTuples()
                self.add_field(name, ncomps, size, file_index=fi)

    def datasets(self):
        """Iterate over all datasets in this source."""
        return (
            ((i,j), self.dataset(i,j))
            for i in range(len(self.paths))
            for j in range(len(self.files))
        )

    def filename(self, path_index, file_index):
        """Return the full file name of a given path and file index."""
        return join(self.paths[path_index], self.files[file_index])

    def dataset(self, path_index, file_index):
        """Return a single dataset associated with a path and file index."""
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(self.filename(path_index, file_index))
        reader.Update()
        return reader.GetOutput()

    def field_mass_matrix(self, field):
        """Return the mass matrix for a single field."""

        # See ramos.utils.vtk.mass_matrix for more info
        return mass_matrix(self.dataset(0, field.file_index), self.variates)

    def field_coefficients(self, field, level=0):
        """Return the coefficient vector for a single field at a given time level."""
        dataset = self.dataset(level, field.file_index)
        pointdata = dataset.GetPointData()
        array = pointdata.GetAbstractArray(field.name)
        return vtk_to_numpy(array)

    def tesselate(self, field, variates=None, level=0, condition=None):
        """Return a tesselation (for plotting) of a single field at a given time level.

        Returns a tuple (x, y, cell_indices), coeffs
        """
        field = self.field(field)
        dataset = self.dataset(level, field.file_index)
        points = vtk_to_numpy(dataset.GetPoints().GetData())

        if not variates:
            variates = self.variates[:2]
        x, y = (points[...,i] for i in variates)
        coeffs = vtk_to_numpy(dataset.GetPointData().GetAbstractArray(field.name))

        cell_indices = get_cell_indices(dataset)

        if condition:
            condition = np.where(points[...,condition] > 0)[0]
            coeffs = coeffs[condition,:]
        x, y, cell_indices = mesh_filter(x, y, cell_indices, condition)

        return (x, y, cell_indices), coeffs

    def sink(self, *args, **kwargs):
        """Create a data sink (an output class) that matches the source."""
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

    def __exit__(self, type_, value, backtrace):
        pass

    def add_level(self, time):
        path = join(self.path, str(time))
        if not exists(path):
            makedirs(path)
        if not isdir(self.path):
            raise IOError
        self.paths.append(path)

    def filename(self, path_index, file_index):
        return join(self.paths[path_index], self.parent.files[file_index])

    def write_fields(self, level, coeffs, fields):
        fields = [self.parent.field(f) for f in fields]
        data = list(zip(fields, decompose(fields, coeffs)))

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

            write_to_file(dataset, self.filename(level, file_index))
