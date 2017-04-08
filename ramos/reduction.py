from itertools import combinations_with_replacement
import logging
import numpy as np

from ramos.utils.parallel import parmap
from ramos.utils.parallel.workers import energy_content, normalized_coeffs, correlation


class Reduction:

    def __init__(self, sources, fields, sink, output, min_modes=10, error=0.05):
        self.sources = sources
        self.master = sources[0].clone(clear_cache=True)
        self.fields = fields
        self.sink = sink
        self.output = output
        self.min_modes = min_modes
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

        logging.info('Computing single-component mass matrices')
        mass = {}
        for field in self.fields:
            mass[field] = self.master.mass_matrix(field, single=True)

        logging.info('Normalizing ensemble')
        args = self.source_levels()
        ensemble = parmap(normalized_coeffs, args, (self.fields, mass))

        logging.info('Computing master mass matrix')
        mass = self.master.mass_matrix(self.fields)

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
            '%d modes suffice for %.2f%% error (threshold %.2f%%)',
            nmodes, 100*actual_error, 100*self.error
        )

        nmodes = min(len(eigvals), max(nmodes, self.min_modes))
        logging.info('Writing %d modes', nmodes)
        with self.sink as sink:
            for i in range(nmodes):
                sink.add_level(i)
                mode = np.zeros(ensemble[0].shape)
                for j, e in enumerate(ensemble):
                    mode += eigvecs[j,i] * e
                mode /= np.sqrt(eigvals[i])
                sink.write_fields(i, mode, self.fields)

        with open(self.output + '.csv', 'w') as f:
            for i, ev in enumerate(eigvals):
                s = np.sum(eigvals[i+1:]) / scale
                f.write('{} {} {} {}\n'.format(
                    i+1, ev, s, np.sqrt(s)
                ))

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
