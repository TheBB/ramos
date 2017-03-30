import click
import logging

from ramos import io
from ramos.reduction import Reduction


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
@click.argument('data', type=io.DataFileType())
def summary(data):
    print(data)


@main.command()
@click.option('--fields', '-f', type=str, multiple=True)
@click.option('--error', '-e', type=float, default=0.05)
@click.argument('sources', type=io.DataFileType(), nargs=-1)
def reduce(fields, error, sources):
    r = Reduction(sources, fields, error)
    r.reduce()


if __name__ == '__main__':
    main()
