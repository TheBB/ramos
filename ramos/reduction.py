from itertools import combinations_with_replacement
import logging
import numpy as np

from ramos.utils.parallel import parmap
from ramos.utils.parallel.workers import energy_content, normalized_coeffs, correlation


class Reduction:

    def __init__(self, sources, fields, error=0.05):
        self.sources = sources
        self.master = sources[0].clone(clear_cache=True)
        self.fields = fields
        self.error = error

    def source_levels(self):
        return [(source, level) for source in self.sources for level in source.levels()]

    @property
    def nsnaps(self):
        if not hasattr(self, '_nsnaps'):
            self._nsnaps = len(self.source_levels())
        return self._nsnaps

    @property
    def nfields(self):
        return len(self.fields)

    def reduce(self):
        self.compute_scales()

        logging.info('Computing master mass matrix')
        mass = self.master.mass_matrix(list(zip(self.fields, self.scales)))
        logging.debug('Mass matrix is %d × %d', *mass.shape)

        logging.info('Normalizing ensemble')
        args = self.source_levels()
        ensemble = parmap(normalized_coeffs, args, (self.fields, mass, mass.sum()))

        logging.info('Computing correlation matrix')
        args = list(combinations_with_replacement(ensemble, 2))
        corrs = parmap(correlation, args, (mass,))
        corrmx = np.empty((self.nsnaps, self.nsnaps))
        i, j = np.triu_indices(self.nsnaps)
        corrmx[i, j] = corrs
        corrmx[j, i] = corrs
        del corrs

        logging.info('Computing eigenvalue decomposition')
        eigvals, eigvecs = np.linalg.eigh(corrmx)
        scale = sum(eigvals)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:,::-1]

        threshold = (1 - self.error ** 2) * scale
        nmodes = min(np.where(np.cumsum(eigvals) > threshold)[0]) + 1
        actual_error = np.sqrt(np.sum(eigvals[nmodes:]) / scale)
        logging.info(
            '%d modes suffice for %.2f%% (threshold %.2f%%)',
            nmodes, 100*actual_error, 100*self.error
        )
        print(eigvals[:10] / scale)

    def compute_scales(self):
        if self.nfields == 1:
            self.scales = np.array([1.0])
            return

        logging.info('Multiple fields, computing scaling factors')
        energies = []
        for field in self.fields:
            logging.debug('Field: %s', field)
            mass = self.master.mass_matrix([field])
            args = self.source_levels()
            energy = parmap(energy_content, args, (field, mass), reduction=sum)
            logging.debug('Energy: %e', energy)
            energies.append(energy)
        scales = 1 / np.array(energies)
        self.scales /= np.sum(scales)
        logging.debug(
            'Scaling factors: %s',
            ', '.join(('{}={}'.format(f, s) for f, s in zip(self.fields, self.scales)))
        )
