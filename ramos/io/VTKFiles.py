from os import makedirs
from os.path import exists, isdir, join, split
from vtk import vtkDataSetReader
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

from ramos.io.Base import DataSource, DataSink
from ramos.utils.mesh import mesh_filter
from ramos.utils.vectors import decompose
from ramos.utils.vtk import mass_matrix, write_to_file, get_cell_indices


class VTKFilesSource(DataSource):

    def __init__(self, files):
        """VTKFilesSource reads this type of structure:

        <basename>-0.vtk
        <basename>-1.vtk
        <basename>-2.vtk
        ...
        <basename>-n.vtk
        """
        self.files = files

        # Try to figure out how many parametric dimensions this data has.
        # Do this by loading any dataset and inspecting its bounding box.
        # (Not foolproof.)
        dataset = self.dataset(0)
        xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()
        variates = [
            abs(a - b) > 1e-5
            for a, b in ((xmin,xmax), (ymin,ymax), (zmin,zmax))
        ]
        pardim = sum(variates)

        # Call superclass constructor
        super(VTKFilesSource, self).__init__(pardim, len(files))

        # List of physical dimensions that are also parametric dimensions
        self.variates = [i for i, v in enumerate(variates) if v]

        # Discover all the fields, by inspecting the files in the first time
        # level. No checking is done to make sure this info is consistent with
        # other time levels.
        pointdata = dataset.GetPointData()
        for i in range(pointdata.GetNumberOfArrays()):
            name = pointdata.GetArrayName(i)
            ncomps = pointdata.GetAbstractArray(i).GetNumberOfComponents()
            size = pointdata.GetAbstractArray(i).GetNumberOfTuples()
            self.add_field(name, ncomps, size)

    def datasets():
        """Iterate over all datasets in this source."""
        return ((i, self.dataset(i)) for i in range(len(self.files)))

    def dataset(self, index):
        """Return a single dataset associated with a file index."""
        reader = vtkDataSetReader()
        reader.SetFileName(self.files[index])
        reader.Update()
        return reader.GetOutput()

    def field_mass_matrix(self, field):
        """Return the mass matrix for a single field."""

        # See ramos.utils.vtk.mass_matrix for more info
        return mass_matrix(self.dataset(0), self.variates)

    def field_coefficients(self, field, level=0):
        """Return the coefficient vector for a single field at a given time level."""
        dataset = self.dataset(level)
        pointdata = dataset.GetPointData()
        array = pointdata.GetAbstractArray(field.name)
        return vtk_to_numpy(array)

    def tesselate(self, field, variates=None, level=0, condition=None):
        """Return a tesselation (for plotting) of a single field at a given time level.

        Returns either:
        - (x, y, cell_indices), coeffs  (if all cells are triangles)
        - (x, y), coeffs                (if there are higher order cells)
        """
        field = self.field(field)
        dataset = self.dataset(level)
        points = vtk_to_numpy(dataset.GetPoints().GetData())

        if not variates:
            variates = self.variates[:2]
        x, y = (points[...,i] for i in self.variates)
        coeffs = vtk_to_numpy(dataset.GetPointData().GetAbstractArray(field.name))

        cell_indices = get_cell_indices(dataset)

        if condition:
            condition = np.where(points[...,condition] > 0)[0]
            coeffs = coeffs[condition,:]
        x, y, cell_indices = mesh_filter(x, y, cell_indices, condition)

        if cell_indices.shape[1] > 3:
            return (x, y), coeffs
        return (x, y, cell_indices), coeffs

    def sink(self, *args, **kwargs):
        """Create a data sink (an output class) that matches the source."""
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
