from argparse import ArgumentParser, RawDescriptionHelpFormatter
import click
from collections import namedtuple, OrderedDict
import importlib
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
@click.argument('filename', type=str)
def structure(filename, out, nx, ny, nz, xval, yval, zval, xmin, ymin, zmin, xmax, ymax, zmax):
    """Turn an unstructured VTK into a structured one."""
    xs = [xval, xval] if xval is not None else [xmin, xmax]
    ys = [yval, yval] if yval is not None else [ymin, ymax]
    zs = [zval, zval] if zval is not None else [zmin, zmax]
    tools = importlib.import_module('gmesh.tools')
    tools.structure(filename, out, [xs, ys, zs], [nx, ny, nz])


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
