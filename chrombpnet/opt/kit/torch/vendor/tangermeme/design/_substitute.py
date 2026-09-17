# _substitute.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numba


@numba.njit(parallel=True, cache=True)
def _fast_tile_substitute(X, motif, idxs):
	"""This function takes a motif and inserts it at all possibilities"""
	
	n_alphabet, n_len = motif.shape
	for i in numba.prange(X.shape[0]):
		idx = idxs[i]
		for j in range(n_len):
			for k in range(n_alphabet):
				X[i, k, j+idx] = motif[k, j]
