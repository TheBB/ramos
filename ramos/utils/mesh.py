import numpy as np


def mesh_filter(x, y, cell_indices, cond=None):
    """Filters the mesh given by x, y and cell_indices according to cond (an array
    of valid indices). Returns a triangular mesh.
    """
    if cond is None:
        cond = np.arange(x.shape[0])
        ind_conv = np.arange(x.shape[0])
        test = set(ind_conv)
    else:
        ind_conv = np.zeros(x.shape, dtype=np.int)
        ind_conv[cond] = np.arange(len(cond))
        test = set(cond)

    final_indices = []
    for cell in cell_indices:
        cell = [ind_conv[c] for c in cell if c in test]
        new_cells = [cell[k:k+3] for k in range(0, len(cell) - 2)]
        final_indices.extend(new_cells)

    return x[cond], y[cond], np.array(final_indices)
