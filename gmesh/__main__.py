from argparse import ArgumentParser, RawDescriptionHelpFormatter
import click
from collections import namedtuple, OrderedDict
import importlib
from os.path import dirname, isfile, split, splitext
import sys
import textwrap


@click.group()
def main():
    pass


@main.command()
@click.option('--zval', type=float)
@click.option('--yval', type=float)
@click.option('--xval', type=float)
@click.option('--zmin', type=float)
@click.option('--ymin', type=float)
@click.option('--xmin', type=float)
@click.option('--zmax', type=float)
@click.option('--ymax', type=float)
@click.option('--xmax', type=float)
@click.option('--nz', type=int, default=10)
@click.option('--ny', type=int, default=10)
@click.option('--nx', type=int, default=10)
@click.option('--out', type=str, required=True)
@click.option('--fprefix', type=str, default='')
@click.option('--timedirs/--no-timedirs', default=False)
@click.argument('filenames', type=str, nargs=-1)
def structure(filenames, timedirs, out, fprefix, nx, ny, nz,
              xval, yval, zval, xmin, ymin, zmin, xmax, ymax, zmax):
    """Turn an unstructured VTK into a structured one."""
    xs = [xval, xval] if xval is not None else [xmin, xmax]
    ys = [yval, yval] if yval is not None else [ymin, ymax]
    zs = [zval, zval] if zval is not None else [zmin, zmax]
    tools = importlib.import_module('gmesh.tools')

    t_start, t_end, ntimes = 0.0, 0.0, 1

    if timedirs:
        files = {}
        for fn in filenames:
            time = split(dirname(fn))[-1]
            files.setdefault(time, []).append(fn)
        files = sorted(list(files.items()), key=lambda k: float(k[0]))
        t_start = float(files[0][0])
        t_end = float(files[-1][0])
        ntimes = len(files)
        first = True
        for level, (time, fns) in enumerate(files):
            for fn in fns:
                print('Level', level, fn, '->', out)
                tools.structure(fn, out, [xs, ys, zs], [nx, ny, nz],
                                level=level, store_basis=first,
                                fprefix=fprefix)
                first = False

        basename, ext = splitext(out)
        if ext in {'.hdf5', '.h5'}:
            data = importlib.import_module('gmesh.data')
            f = data.IFEMFile(out)
            f.set_timestep((t_end - t_start) / (ntimes - 1))

    else:
        assert len(filenames) == 1
        print(filenames[0], '->', out)
        tools.structure(filenames[0], out, [xs, ys, zs], [nx, ny, nz], fprefix=fprefix)


@main.command()
@click.option('--fields', '-f', type=str, multiple=True)
@click.argument('filenames', type=str, nargs=-1)
def reduce(fields, filenames):
    """Dimensional reduction analysis."""
    tools = importlib.import_module('gmesh.tools')
    tools.reduce(fields, filenames)


@main.command()
@click.option('--level', '-l', type=int, default=0)
@click.argument('filename', type=str)
@click.argument('field', type=str)
def plot(filename, field, level):
    comp = 0
    if ':' in field:
        field, comp = field.split(':')
        if comp != 'ss':
            comp = int(comp)
    tools = importlib.import_module('gmesh.tools')
    tools.plot(filename, field, comp, level)


@main.command()
def map():
    """Show the map."""
    gui = importlib.import_module('gmesh.gui')
    gui.run()


if __name__ == '__main__':
    main()
