from . import data


def structure(fn):
    f = next(data.read(fn))
    print(f)
