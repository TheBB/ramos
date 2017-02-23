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
@click.option('--out', type=str, required=False)
@click.option('--normal', type=float, nargs=3, required=False)
@click.option('--base', type=float, nargs=3, required=False)
@click.option('--auto/--no-auto', default=False)
@click.argument('filenames', type=str, nargs=-1)
def transform(filenames, normal, base, auto, out):
    tools = importlib.import_module('gmesh.tools')

    for f in filenames:
        this_out = out
        if this_out is None:
            bb, ext = splitext(f)
            this_out = '{}_trf{}'.format(bb, ext)
        tools.transform(f, normal, base, auto, this_out)


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
@click.option('--tolerance', type=float, default=None)
@click.option('--step', type=int, default=1)
@click.argument('filenames', type=str, nargs=-1)
def structure(filenames, timedirs, step, out, fprefix, nx, ny, nz,
              xval, yval, zval, xmin, ymin, zmin, xmax, ymax, zmax, tolerance):
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
        files = files[::step]
        t_start = float(files[0][0])
        t_end = float(files[-1][0])
        ntimes = len(files)
        first = True
        for level, (time, fns) in enumerate(files):
            for fn in fns:
                print('Level', level, fn, '->', out)
                tools.structure(fn, out, [xs, ys, zs], [nx, ny, nz],
                                level=level, store_basis=first,
                                fprefix=fprefix, tolerance=tolerance)
                first = False

        basename, ext = splitext(out)
        if ext in {'.hdf5', '.h5'}:
            data = importlib.import_module('gmesh.data')
            f = data.IFEMFile(out)
            if ntimes > 1:
                f.set_timestep((t_end - t_start) / (ntimes - 1), t_start, t_end)

    else:
        assert len(filenames) == 1
        print(filenames[0], '->', out)
        tools.structure(filenames[0], out, [xs, ys, zs], [nx, ny, nz], fprefix=fprefix, tolerance=tolerance)


@main.command()
@click.option('--fields', '-f', type=str, multiple=True)
@click.option('--out', type=str, required=True)
@click.argument('filenames', type=str, nargs=-1)
def reduce(fields, filenames, out):
    """Dimensional reduction analysis."""
    tools = importlib.import_module('gmesh.tools')
    tools.reduce(fields, filenames, out)


@main.command()
@click.option('--out', type=str, default=None)
@click.argument('filename', type=str)
def spectrum(filename, out):
    """Dimensional reduction analysis."""
    if out is None:
        basename, _ = splitext(filename)
        out = basename + '.csv'
    tools = importlib.import_module('gmesh.tools')
    tools.spectrum(filename, out)


@main.command()
@click.option('--level', '-l', type=int, default=0)
@click.option('--out', type=str, default=None)
@click.option('--colorbar/--no-colorbar', default=True)
@click.option('--ticks/--no-ticks', default=True)
@click.option('--show/--no-show', default=True)
@click.option('--style', default='imshow')
@click.argument('filename', type=str)
@click.argument('field', type=str)
def plot(filename, field, **kwargs):
    comp = 0
    if ':' in field:
        field, comp = field.split(':')
        try: comp = int(comp)
        except ValueError: pass
    tools = importlib.import_module('gmesh.tools')
    tools.plot(filename, field, comp=comp, **kwargs)


@main.command()
@click.option('--level', '-l', type=int, default=0)
@click.option('--varying', type=str, default=None)
@click.argument('filename', type=str)
@click.argument('field', type=str)
def avg(filename, field, varying, level):
    tools = importlib.import_module('gmesh.tools')
    tools.avg(filename, field, varying, level)


@main.command('disp-flux')
@click.option('--level', '-l', type=int, default=0)
@click.option('--varying', type=str, default=None)
@click.argument('filename', type=str)
def disp_flux(filename, varying, level):
    tools = importlib.import_module('gmesh.tools')
    tools.disp_flux(filename, varying, level)


@main.command()
@click.option('--out', type=str, required=True)
@click.argument('filename', type=str)
@click.argument('field', type=str)
def animate(filename, field, out):
    comp = 0
    if ':' in field:
        field, comp = field.split(':')
        try: comp = int(comp)
        except ValueError: pass
    tools = importlib.import_module('gmesh.tools')
    tools.animate(filename, field, comp, out)


@main.command()
def map():
    """Show the map."""
    gui = importlib.import_module('gmesh.gui')
    gui.run()


if __name__ == '__main__':
    main()
