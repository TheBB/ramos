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
@click.option('--nz', type=int, default=10)
@click.option('--ny', type=int, default=10)
@click.option('--nx', type=int, default=10)
@click.option('--out', type=str, required=True)
@click.argument('filename', type=str)
def structure(filename, out, nx, ny, nz, xval, yval, zval):
    """Turn an unstructured VTK into a structured one."""
    tools = importlib.import_module('gmesh.tools')
    tools.structure(filename, out, nx, ny, nz, xval, yval, zval,)


@main.command()
def map():
    """Show the map."""
    gui = importlib.import_module('gmesh.gui')
    gui.run()


if __name__ == '__main__':
    main()
