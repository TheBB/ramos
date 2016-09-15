import numpy as np
import vtk
import sys
from itertools import product, islice, repeat, tee
from multiprocessing import Pool
from . import data


def structure(fn, nx, ny, nz, out):
    f = next(data.read(fn))
    dataset = f.reader.GetOutput()

    xmin, xmax, ymin, ymax, zmin, zmax = dataset.GetBounds()

    xs = np.linspace(xmin, xmax, nx+1)
    ys = np.linspace(ymin, ymax, ny+1)
    zs = np.linspace(zmin, zmax, nz+1)

    points = vtk.vtkPoints()
    for z, y, x in product(zs, ys, xs):
        points.InsertNextPoint(x, y, z)
    new_grid = vtk.vtkStructuredGrid()
    new_grid.SetDimensions(nx+1, ny+1, nz+1)
    new_grid.SetPoints(points)

    probefilter = vtk.vtkProbeFilter()
    probefilter.SetSourceConnection(f.reader.GetOutputPort())
    probefilter.SetInputData(new_grid)
    probefilter.Update()

    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(out)
    writer.SetInputConnection(probefilter.GetOutputPort())
    writer.Write()
