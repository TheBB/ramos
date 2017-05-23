from itertools import combinations_with_replacement
import logging
import numpy as np

from ramos.utils.parallel import parmap
from ramos.utils.parallel.workers import energy_content, normalized_coeffs, mv_dot, vv_dot


class Reduction:

    def __init__(self, sources, fields, sink, output, min_modes=10, error=0.05):
        """Create a reduced basis using POD.

        - `sources`: The data sources to use as input
        - `fields`: The field names to read
        - `sink`: The data sink to use as output
        - `output`: Name of the csv file to write spectral information to
        - `min_modes`: Minimum number of modes to write
        - `error`: Error threshold to achieve
        """
        self.sources = sources
        self.fields = fields
        self.sink = sink
        self.output = output
        self.min_modes = min_modes
        self.error = error

        # Create a master source that will be used to compute mass matrices.
        # Other sources will be passed to worker processes, so we want to keep
        # them as lightweight as possible. Therefore, we create an identical
        # copy of the first source and give it its own unique mass matrix
        # cache, so that it, and only it, will carry a large amount of data.
        self.master = sources[0].clone(clear_cache=True)

    def source_levels(self):
        """Iterate over all pairs of sources and levels."""
        return [(source, level) for source in self.sources for level in source.levels()]

    @property
    def nsnaps(self):
        """Number of snapshots available."""
        if not hasattr(self, '_nsnaps'):
            self._nsnaps = len(self.source_levels())
        return self._nsnaps

    @property
    def nfields(self):
        """Number of fields under consideration."""
        return len(self.fields)

    def reduce(self):
        """Compute a reduced basis using POD."""

        # If there are multiple fields, we must compute the weight for each of
        # them, so that they have equal energy contribution.
        self.compute_scales()

        # Compute the mass matrices associated with each field and store them
        # in a dict. The DataSource.mass_matrix function is parallelized, so do
        # this in serial.
        logging.info('Computing single-component mass matrices')
        mass = {}
        for field in self.fields:
            mass[field] = self.master.mass_matrix(field, single=True)

        # Compute the coefficients for each snapshot, centered around the
        # component-wise mean.
        logging.info('Normalizing ensemble')
        args = self.source_levels()
        ensemble = parmap(normalized_coeffs, args, (self.fields, mass))

        # Compute the grand unified mass matrix that applies to the grand
        # unified coefficient vectors (with multiple fields). This should be
        # quick since it uses cached data.
        logging.info('Computing master mass matrix')
        mass = self.master.mass_matrix(self.fields)

        # Compute all the matrix-vector products.
        logging.info('Computing matrix-vector products')
        ensemble_m = parmap(mv_dot, ensemble, (mass,), unwrap=False)

        # Compute the actual covariance matrix, made up of terms of the type
        # u^T × M × v, where u and v are coefficient vectors. To do this, we
        # calculate dot-products of all pairs of vectors in ensemble and
        # ensemble_m.
        logging.info('Computing covariance matrix')
        args = [
            (a, b) for (a, _), (_, b) in
            combinations_with_replacement(zip(ensemble, ensemble_m), 2)
        ]
        corrs = parmap(vv_dot, args)
        corrmx = np.empty((self.nsnaps, self.nsnaps))
        i, j = np.triu_indices(self.nsnaps)
        corrmx[i, j] = corrs
        corrmx[j, i] = corrs
        del corrs               # Let GC deal with this one

        # Compute the eigenvalue decomposition of the covariance matrix,
        # ordered from high to low eigenvalues.
        logging.info('Computing eigenvalue decomposition')
        eigvals, eigvecs = np.linalg.eigh(corrmx)
        scale = sum(eigvals)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:,::-1]

        # Compute the number of modes necessary to satisfy the error threshold,
        # and the actual error achieved.
        threshold = (1 - self.error ** 2) * scale
        nmodes = min(np.where(np.cumsum(eigvals) > threshold)[0]) + 1
        actual_error = np.sqrt(np.sum(eigvals[nmodes:]) / scale)
        logging.info(
            '%d modes suffice for %.2f%% error (threshold %.2f%%)',
            nmodes, 100*actual_error, 100*self.error
        )

        # Write modes to sink
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

        # Write spectrum to CSV file
        with open(self.output + '.csv', 'w') as f:
            for i, ev in enumerate(eigvals):
                s = np.sum(eigvals[i+1:]) / scale
                f.write('{} {} {} {}\n'.format(
                    i+1, ev/scale, s, np.sqrt(s)
                ))

    def compute_scales(self):
        """Compute weighing factors for each field."""

        # Trivial case: only one field
        if self.nfields == 1:
            self.scales = np.array([1.0])
            return

        logging.info('Multiple fields, computing scaling factors')

        # Compute the energy content of each field
        energies = []
        for field in self.fields:
            logging.debug('Field: %s', field)
            mass = self.master.mass_matrix([field])
            args = self.source_levels()
            energy = parmap(energy_content, args, (field, mass), reduction=sum)
            logging.debug('Energy: %e', energy)
            energies.append(energy)

        # Compute scaling factors so that the total energy is 1 (this is arbitrary),
        # and each field contributes an equal amount to it.
        scales = 1 / np.array(energies)
        self.scales /= np.sum(scales)
        logging.debug(
            'Scaling factors: %s',
            ', '.join(('{}={}'.format(f, s) for f, s in zip(self.fields, self.scales)))
        )
