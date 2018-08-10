import click
from importlib import import_module
import logging
import numpy as np
from tqdm import tqdm
from vtk import vtkProbeFilter

from ramos import io
from ramos.reduction import Reduction
from ramos.utils.vtk import write_to_file
from ramos.utils.parallel import parmap
from ramos.utils.parallel.workers import mv_dot, vv_dot


@click.group()
@click.option('--verbosity', '-v',
              type=click.Choice(['debug', 'info', 'warning', 'error', 'critical']),
              default='info')
def main(verbosity):
    logging.basicConfig(
        format='{asctime} {levelname: <10} {message}',
        datefmt='%H:%M',
        style='{',
        level=verbosity.upper(),
    )


@main.command()
@click.argument('data', type=io.DataSourceType())
def summary(data):
    """Print a brief summary of a data source."""
    print(data)


@main.command()
@click.option('--field', '-f', 'fields', type=str, multiple=True, help='Fields to read')
@click.option('--error', '-e', type=float, default=0.05, help='Relative error threshold to achieve')
@click.option('--out', '-o', type=str, default='out', help='Name of output')
@click.option('--min-modes', type=int, default=10, help='Minimum number of modes to write')
@click.argument('sources', type=io.DataSourceType(), nargs=-1)
def reduce(fields, error, out, min_modes, sources):
    """Calculate a reduced basis."""
    sink = sources[0].sink(out)
    r = Reduction(sources, fields, sink, out, min_modes, error)
    r.reduce()


@main.command()
@click.option('--target', '-t', type=io.DataSourceType(), help='Basis to project onto')
@click.option('--out', '-o', type=str, default='out', help='Name of output')
@click.argument('source', type=io.DataSourceType())
def project(source, target, out):
    """Project a data source onto a basis."""
    fields = [f.name for f in target.fields()]
    mass = target.mass_matrix(fields)
    sink = source.sink(out)

    modes = [target.coefficients(fields, li) for li in target.levels()]

    # Project each time level individually
    for li in tqdm(source.levels(), desc='Time steps', total=source.ntimes):
        vector = source.coefficients(fields, li)
        coeffs = [mass.dot(mode).dot(vector) for mode in modes]
        recons = np.zeros(vector.shape)
        for coeff, mode in zip(coeffs, modes):
            recons += coeff * mode
        sink.add_level(li)
        sink.write_fields(li, recons, fields)


@main.command()
@click.option('--target', '-t', type=io.DataSourceType(), help='Source from which the mesh will be taken')
@click.option('--out', '-o', type=str, default='out', help='Name of output')
@click.argument('source', type=io.DataSourceType())
def interpolate(source, target, out):
    """Interpolate a data source on a common mesh."""
    if not target:
        target = source

    # Interpolation only works on VTK type sources currently
    assert isinstance(source, (io.VTKFilesSource, io.VTKTimeDirsSource))
    assert isinstance(target, (io.VTKFilesSource, io.VTKTimeDirsSource))
    sink = source.sink(out)
    for i in source.levels():
        sink.add_level(i)

    probefilter = vtkProbeFilter()
    _, dataset = next(target.datasets())
    probefilter.SetInputData(dataset)
    # Depending on the source type, a dataset may or may not correspond to a
    # time level. However, the data sets make up all the information in a
    # source, so dealing with all of them will create a complete copy.
    for ind, ds in tqdm(source.datasets()):
        probefilter.SetSourceData(ds)
        probefilter.Update()
        output = probefilter.GetUnstructuredGridOutput()
        if not output:
            output = probefilter.GetPolyDataOutput()
        if not output:
            raise TypeError('Unsupported dataset type')
        write_to_file(output, sink.filename(*ind))


@main.command()
@click.option('--field', '-f', type=str, help='Field to plot')
@click.option('--level', '-l', type=int, default=0, help='Time level to plot at')
@click.option('--out', '-o', type=str, help='Name of output')
@click.option('--scale/--no-scale', default=False, help='Add a color scale to the output')
@click.option('--smooth/--no-smooth', default=False, help='Smooth shading of mesh')
@click.option('--show/--no-show', default=True, help='Show plot in a window')
@click.option('--transpose/--no-transpose', default=False, help='Transpose x and y')
@click.option('--flip-x/--no-flip-x', default=False, help='Flip x')
@click.option('--flip-y/--no-flip-y', default=False, help='Flip y')
@click.option('--variates', default=(0,1), nargs=2, type=int, help='Directions to plot')
@click.option('--condition', default=None, type=int, help='Pseudo-2D direction for filtering')
@click.option('--xlim', default=None, nargs=2, type=int, help='Bounding box')
@click.option('--ylim', default=None, nargs=2, type=int, help='Bounding box')
@click.option('--cmap', default='viridis', help='Colormap to use')
@click.argument('source', type=io.DataSourceType())
def plot(field, level, out, scale, smooth, show,
         transpose, flip_x, flip_y, variates, condition,
         xlim, ylim, cmap, source):
    """Plot data from a data source."""

    # We suport some minor post-processing of fields, in the form of
    # <fieldname>:<postproc>
    if ':' in field:
        field, post = field.split(':')
    else:
        post = None

    # tess is either (x, y) or (x, y, cell_indices)
    tess, coeffs = source.tesselate(field, variates, level, condition)

    # Norm or norm squared
    if post in {'ss', 'ssq'}:
        coeffs = np.sum(coeffs ** 2, axis=-1)
        if post == 'ssq':
            coeffs = np.sqrt(coeffs)
    # Pick a single component
    elif post and post in 'xyz':
        coeffs = coeffs[..., 'xyz'.index(post)]
    elif post:
        coeffs = coeffs[..., int(post)]
    else:
        coeffs = coeffs[..., 0]

    # Import here to ensure that matplotlib is an optional dependency
    mpl = import_module('matplotlib')
    plt = import_module('matplotlib.pyplot')
    Triangulation = import_module('matplotlib.tri').Triangulation

    # Make some changes to x and y depending on input
    x, y = tess[0], tess[1]
    if transpose: x, y = y, x
    if flip_x: x = -x
    if flip_y: y = -y
    tess = tuple([x, y] + list(tess[2:]))

    # Form triangulation and make the plot
    # If tess is only (x, y), the Delaunay triangulation is used
    tri = Triangulation(*tess)
    plt.tripcolor(tri, coeffs, shading=('gouraud' if smooth else 'flat'), cmap=plt.get_cmap(cmap))

    plt.axes().set_aspect(1)
    plt.axes().axis('off')
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if scale:
        plt.colorbar()
    if out:
        plt.savefig(out, bbox_inches='tight', pad_inches=0, dpi=300)
    if show:
        plt.show()


if __name__ == '__main__':
    main()
