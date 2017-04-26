import vtk
import numpy as np

from ramos.utils.vtk import mass_matrix
from ramos.utils.matrix import MatrixBuilder


def test_vtk_quads():
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(2.0, 0.0, 0.0)
    points.InsertNextPoint(0.0, 1.0, 0.0)
    points.InsertNextPoint(1.0, 1.0, 0.0)
    points.InsertNextPoint(2.0, 1.0, 0.0)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(vtk.VTK_QUAD, 4, [0, 3, 4, 1])
    grid.InsertNextCell(vtk.VTK_QUAD, 4, [1, 4, 5, 2])
    builder = MatrixBuilder()
    data, rows, cols = mass_matrix(grid, [0, 1], parallel=False)
    builder.add(data, rows, cols, 1)
    assert np.allclose(
        builder.build().toarray(),
        np.array([
            [1/9, 1/18, 0, 1/18, 1/36, 0],
            [1/18, 2/9, 1/18, 1/36, 1/9, 1/36],
            [0, 1/18, 1/9, 0, 1/36, 1/18],
            [1/18, 1/36, 0, 1/9, 1/18, 0],
            [1/36, 1/9, 1/36, 1/18, 2/9, 1/18],
            [0, 1/36, 1/18, 0, 1/18, 1/9],
        ])
    )


def test_vtk_quads_stretched():
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(3.0, 0.0, 0.0)
    points.InsertNextPoint(0.0, 1.0, 0.0)
    points.InsertNextPoint(1.0, 1.0, 0.0)
    points.InsertNextPoint(3.0, 1.0, 0.0)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(vtk.VTK_QUAD, 4, [0, 3, 4, 1])
    grid.InsertNextCell(vtk.VTK_QUAD, 4, [1, 4, 5, 2])
    builder = MatrixBuilder()
    data, rows, cols = mass_matrix(grid, [0, 1], parallel=False)
    builder.add(data, rows, cols, 1)
    assert np.allclose(
        builder.build().toarray(),
        np.array([
            [1/9, 1/18, 0, 1/18, 1/36, 0],
            [1/18, 3/9, 1/9, 1/36, 1/6, 1/18],
            [0, 1/9, 2/9, 0, 1/18, 1/9],
            [1/18, 1/36, 0, 1/9, 1/18, 0],
            [1/36, 1/6, 1/18, 1/18, 3/9, 1/9],
            [0, 1/18, 1/9, 0, 1/9, 2/9],
        ])
    )


def test_vtk_triangles():
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(0.0, 1.0, 0.0)
    points.InsertNextPoint(1.0, 1.0, 0.0)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, [0, 2, 1])
    grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, [1, 2, 3])
    builder = MatrixBuilder()
    data, rows, cols = mass_matrix(grid, [0, 1], parallel=False)
    builder.add(data, rows, cols, 1)
    assert np.allclose(
        builder.build().toarray(),
        np.array([
            [1/12, 1/24, 1/24, 0],
            [1/24, 1/6, 1/12, 1/24],
            [1/24, 1/12, 1/6, 1/24],
            [0, 1/24, 1/24, 1/12],
        ])
    )
