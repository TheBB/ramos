import click
from itertools import product, combinations_with_replacement
import logging
import numpy as np

from ramos import io
from ramos.utils.parallel import parmap
from ramos.utils.parallel.workers import energy_content, normalized_coeffs, correlation


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
    master = sources[0].clone(clear_cache=True)

    if len(fields) > 1:
        logging.info('Multiple fields, computing scaling factors')
        energies = []
        for field in fields:
            logging.debug('Field: %s', field)
            mass = master.mass_matrix([field])
            args = [(source, level) for source in sources for level in source.levels()]
            energy = parmap(energy_content, args, (field, mass), reduction=sum)
            logging.debug('Energy: %e', energy)
            energies.append(energy)
        scales = 1 / np.array(energies)
        scales /= np.sum(scales)
        logging.debug(
            'Scaling factors: %s',
            ', '.join(('{}={}'.format(f, s) for f, s in zip(fields, scales)))
        )
    else:
        scales = np.array([1.0])

    logging.info('Computing master mass matrix')
    mass = master.mass_matrix(list(zip(fields, scales)))

    logging.info('Normalizing ensemble')
    args = [(source, level) for source in sources for level in source.levels()]
    nsnaps = len(args)
    area = mass.sum()
    ensemble = parmap(normalized_coeffs, args, (fields, mass, area))

    logging.info('Computing correlation matrix')
    args = list(combinations_with_replacement(ensemble, 2))
    corrs = parmap(correlation, args, (mass,))
    corrmx = np.empty((nsnaps, nsnaps))
    i, j = np.triu_indices(nsnaps)
    corrmx[i, j] = corrs
    corrmx[j, i] = corrs
    del corrs

    logging.info('Computing eigenvalue decomposition')
    eigvals, eigvecs = np.linalg.eigh(corrmx)
    scale = sum(eigvals)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:,::-1]

    threshold = (1 - error ** 2) * scale
    nmodes = min(np.where(np.cumsum(eigvals) > threshold)[0]) + 1
    actual_error = np.sqrt(np.sum(eigvals[nmodes:]) / scale)
    logging.info(
        '%d modes suffice for %.2f%% (threshold %.2f%%)',
        nmodes, 100*actual_error, 100*error
    )
    print(eigvals[:10] / scale)


if __name__ == '__main__':
    main()
