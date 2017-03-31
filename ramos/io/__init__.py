import click
from operator import itemgetter
from os.path import abspath, dirname, exists, isdir, isfile, join, splitext
from os import listdir

from ramos.io.Base import DataSource
from ramos.io.IFEMFile import IFEMFileSource
from ramos.io.VTKFiles import VTKFilesSource
from ramos.io.VTKTimeDirs import VTKTimeDirsSource


__all__ = ['load', 'DataSourceType']


def vtk_split(filename):
    basename, ext = splitext(filename)
    if ext != '.vtk' or '-' not in basename:
        raise ValueError()
    level, base = (s[::-1] for s in basename[::-1].split('-', maxsplit=1))
    return base, int(level)


def _load_dir(path, fields):
    files, dirs = {}, {}
    for sub in listdir(path):
        full = join(path, sub)

        try:
            assert isfile(full)
            base, level = vtk_split(sub)
            files.setdefault(base, {})[level] = full
        except (AssertionError, ValueError):
            pass

        try:
            assert isdir(full)
            dirs[float(sub)] = full
        except (AssertionError, ValueError):
            pass

    files = [
        [v[i] for i in range(len(v))]
        for v in files.values()
        if all(i in v for i in range(len(v)))
    ]
    try:
        files = max(files, key=len)
    except ValueError:
        files = None

    if dirs and (not files or len(dirs) > len(files)):
        dirs = [d for _, d in sorted(dirs.items(), key=itemgetter(0))]
        return VTKTimeDirsSource(dirs)
    elif files:
        return VTKFilesSource(files)
    else:
        print('Nothing')


def _load_file(filename, fields):
    basename, ext = splitext(filename)

    if ext == '.mat':
        dep_files = ['{}-lr{}.lr'.format(basename, field) for field in fields]
        if not all(exists(f) for f in dep_files):
            raise FileNotFoundError()
        return DataSource(filename)

    if ext in {'.hdf5', '.h5'}:
        dep_file = '{}.xml'.format(basename)
        if not exists(dep_file):
            raise FileNotFoundError()
        return IFEMFileSource(filename)


def load(filename, fields=[]):
    if not exists(filename):
        raise FileNotFoundError()
    if isdir(filename):
        obj = _load_dir(filename, fields)
    else:
        obj = _load_file(filename, fields)
    if not obj:
        raise FileNotFoundError()
    return obj


class DataSourceType(click.ParamType):
    name = 'data'

    def convert(self, value, param, ctx):
        try:
            return load(value)
        except FileNotFoundError:
            self.fail('{} is not a valid data location'.format(value), param, ctx)
