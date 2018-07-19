import click
from operator import itemgetter
from os.path import abspath, dirname, exists, isdir, isfile, join, splitext
from os import listdir

from ramos.io.Base import DataSource
from ramos.io.VTKFiles import VTKFilesSource
from ramos.io.VTKTimeDirs import VTKTimeDirsSource

try:
    import splipy
    from ramos.io.IFEMFile import IFEMFileSource
    has_ifem = True
except ImportError:
    has_ifem = False


__all__ = ['load', 'DataSourceType']


def vtk_split(filename):
    """Splits a vtk filename xxxx-nnn.vtk into base (xxxx) and level (nnn)."""
    basename, ext = splitext(filename)
    if ext != '.vtk' or '-' not in basename:
        raise ValueError()
    level, base = (s[::-1] for s in basename[::-1].split('-', maxsplit=1))
    return base, int(level)


def _load_dir(path, fields):
    """Load a data source from the directory given by `path`. Optionally give a
    list of fields to load (only valid for some types of data sources).
    """

    # Look through the contents of the directory, for files and subdirectories
    files, dirs = {}, {}
    for sub in listdir(path):
        full = join(path, sub)

        try:
            # If it's a file, check if it's on the xxxx-nnn.vtk form. If it is,
            # add it to the files dict.
            assert isfile(full)
            base, level = vtk_split(sub)
            files.setdefault(base, {})[level] = full
        except (AssertionError, ValueError):
            pass

        try:
            # If it's a directory, check if its name is a valid floating point
            # number. If it is, add it to the dirs dict.
            assert isdir(full)
            dirs[float(sub)] = full
        except (AssertionError, ValueError):
            pass

    # At this point, files is a dict mapping base names (xxxx) to dicts, which
    # again map levels (nnn) to file names, while dirs is a list mapping times
    # to paths.

    # Remove all entries in files which have "holes" in them. An entry is valid
    # if all levels 0, 1, ..., n are present.
    files = [
        [v[i] for i in range(len(v))]
        for v in files.values()
        if all(i in v for i in range(len(v)))
    ]

    # If there are more than one entry in files (i.e. more than one valid base
    # name), the one that is longest is assumed to be the correct one.
    try:
        files = max(files, key=len)
    except ValueError:
        files = None

    # If we found subdirectories, and there are more subdirectories than files,
    # we assume that the user wants the subdirectories.
    if dirs and (not files or len(dirs) > len(files)):
        # Sort by time and create a VTKTimeDirsSource object
        dirs = [d for _, d in sorted(dirs.items(), key=itemgetter(0))]
        return VTKTimeDirsSource(dirs)
    elif files:
        # Otherwise, create a VTKFilesSource object
        return VTKFilesSource(files)


def _load_file(filename, fields):
    """Load a data source from the file given by `filename`. Optionally give a
    list of fields to load (only valid for some types of data sources).
    """
    basename, ext = splitext(filename)

    # Kjetil's MATLAB LR-spline files (not finished)
    if ext == '.mat':
        dep_files = ['{}-lr{}.lr'.format(basename, field) for field in fields]
        if not all(exists(f) for f in dep_files):
            raise FileNotFoundError()
        return DataSource(filename)

    # IFEM results
    if ext in {'.hdf5', '.h5'} and has_ifem:
        dep_file = '{}.xml'.format(basename)
        if not exists(dep_file):
            raise FileNotFoundError()
        return IFEMFileSource(filename)


def load(filename, fields=[]):
    """Load a data source from the location given by `filename`. Optionally give a
    list of fields to load (only valid for some types of data sources).
    """
    if not exists(filename):
        raise FileNotFoundError()
    # Dispatch to _load_dir for directories, or _load_file for filenames.
    if isdir(filename):
        obj = _load_dir(filename, fields)
    else:
        obj = _load_file(filename, fields)
    # _load_dir or _load_file may return None
    if not obj:
        raise FileNotFoundError()
    return obj


class DataSourceType(click.ParamType):
    """Parameter type for data sources which can be used as the type argument in
    click options or arguments.
    """
    name = 'data'

    def convert(self, value, param, ctx):
        try:
            return load(value)
        except FileNotFoundError:
            self.fail('{} is not a valid data location'.format(value), param, ctx)
