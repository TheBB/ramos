import click
from importlib import import_module
import logging
import numpy as np
from tqdm import tqdm
from vtk import vtkProbeFilter

from ramos import io
from ramos.reduction import Reduction
from ramos.utils.vtk import write_to_file


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
    print(data)


@main.command()
@click.option('--fields', '-f', type=str, multiple=True)
@click.option('--error', '-e', type=float, default=0.05)
@click.option('--out', '-o', type=str, default='out')
@click.option('--min-modes', type=int, default=10)
@click.argument('sources', type=io.DataSourceType(), nargs=-1)
def reduce(fields, error, out, min_modes, sources):
    sink = sources[0].sink(out)
    r = Reduction(sources, fields, sink, out, min_modes, error)
    r.reduce()


@main.command()
@click.option('--target', '-t', type=io.DataSourceType())
@click.option('--out', '-o', type=str, default='out')
@click.argument('source', type=io.DataSourceType())
def interpolate(source, target, out):
    assert isinstance(source, (io.VTKFilesSource, io.VTKTimeDirsSource))
    assert isinstance(target, (io.VTKFilesSource, io.VTKTimeDirsSource))
    sink = source.sink(out)
    for i in source.levels():
        sink.add_level(i)

    probefilter = vtkProbeFilter()
    _, dataset = next(target.datasets())
    probefilter.SetInputData(dataset)
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
@click.option('--field', '-f', type=str)
@click.option('--level', '-l', type=int, default=0)
@click.option('--out', '-o', type=str)
@click.option('--scale/--no-scale', default=False)
@click.option('--smooth/--no-smooth', default=False)
@click.option('--show/--no-show', default=False)
@click.option('--transpose/--no-transpose', default=False)
@click.option('--flip-x/--no-flip-x', default=False)
@click.option('--flip-y/--no-flip-y', default=False)
@click.option('--cmap', default='viridis')
@click.argument('source', type=io.DataSourceType())
def plot(field, level, out, scale, smooth, show, transpose, flip_x, flip_y, source, cmap):
    assert source.pardim == 2
    if ':' in field:
        field, post = field.split(':')
    else:
        post = None
    tess, coeffs = source.tesselate(field, level)
    if post in {'ss', 'ssq'}:
        coeffs = np.sum(coeffs ** 2, axis=-1)
        if post == 'ssq':
            coeffs = np.sqrt(coeffs)
    elif post and post in 'xyz':
        coeffs = coeffs[..., 'xyz'.index(post)]
    elif post:
        coeffs = coeffs[..., int(post)]
    else:
        coeffs = coeffs[..., 0]

    mpl = import_module('matplotlib')
    plt = import_module('matplotlib.pyplot')
    Triangulation = import_module('matplotlib.tri').Triangulation

    x, y = tess[0], tess[1]
    if transpose: x, y = y, x
    if flip_x: x = -x
    if flip_y: y = -y
    tess = tuple([x, y] + list(tess[2:]))

    tri = Triangulation(*tess)
    plt.tripcolor(tri, coeffs, shading=('gouraud' if smooth else 'flat'), cmap=plt.get_cmap(cmap))

    plt.axes().set_aspect(1)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    if scale:
        plt.colorbar()
    if out:
        plt.savefig(out, bbox_inches='tight', pad_inches=0, dpi=300)
    if show:
        plt.show()


if __name__ == '__main__':
    main()
