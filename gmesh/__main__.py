from argparse import ArgumentParser, RawDescriptionHelpFormatter
import click
from collections import namedtuple, OrderedDict
import importlib
from os.path import dirname, isfile, splitext
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
@click.option('--timedirs/--no-timedirs', default=False)
@click.argument('filenames', type=str, nargs=-1)
def structure(filenames, timedirs, out, nx, ny, nz,
              xval, yval, zval, xmin, ymin, zmin, xmax, ymax, zmax):
    """Turn an unstructured VTK into a structured one."""
    xs = [xval, xval] if xval is not None else [xmin, xmax]
    ys = [yval, yval] if yval is not None else [ymin, ymax]
    zs = [zval, zval] if zval is not None else [zmin, zmax]
    tools = importlib.import_module('gmesh.tools')

    fields = {}
    t_start, t_end, ntimes = 0.0, 0.0, 1

    if timedirs:
        files = {}
        for fn in filenames:
            time = dirname(fn)
            files.setdefault(time, []).append(fn)
        files = sorted(list(files.items()), key=lambda k: float(k[0]))
        t_start = float(files[0][0])
        t_end = float(files[-1][0])
        ntimes = len(files)
        first = True
        for level, (time, fns) in enumerate(files):
            for fn in fns:
                print('Level', level, fn, '->', out)
                fields.update(tools.structure(fn, out, [xs, ys, zs], [nx, ny, nz],
                                              level=level, store_basis=first))
                first = False
    else:
        assert len(filenames) == 1
        print(filenames[0], '->', out)
        tools.structure(filenames[0], out, [xs, ys, zs], [nx, ny, nz])

    basename, ext = splitext(out)
    if ext == '.hdf5' and fields:
        with open(basename + '.xml', 'w') as f:
            f.write('<stuff>\n')
            f.write('  <levels>{}</levels>\n'.format(ntimes))
            for fname, coefs in fields.items():
                f.write('  <entry type="field" name="{}" basis="basis" components="{}" />\n'.format(
                    fname, coefs.shape[-1]
                ))
            f.write('  <timestep constant="1" order="1" interval="1">{}</timestep>\n'.format(
                (t_end - t_start) / (ntimes - 1)
            ))
            f.write('</stuff>\n')


@main.command()
@click.option('--fields', '-f', type=str, multiple=True)
@click.argument('filenames', type=str, nargs=-1)
def reduce(fields, filenames):
    """Dimensional reduction analysis."""
    tools = importlib.import_module('gmesh.tools')
    tools.reduce(fields, filenames)


@main.command()
def map():
    """Show the map."""
    gui = importlib.import_module('gmesh.gui')
    gui.run()


if __name__ == '__main__':
    main()
