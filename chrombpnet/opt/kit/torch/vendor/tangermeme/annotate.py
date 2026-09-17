# annotate.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

from typing import Any

import numpy
import torch
import pandas

from .io import read_meme
from .utils import _validate_input

from memelite.tomtom import tomtom


def annotate_seqlets(
	X: torch.Tensor,
	seqlets: pandas.DataFrame,
	motifs: str | dict,
	n_nearest: int = 1,
	n_jobs: int = -1,
	**kwargs: Any,
) -> tuple[pandas.DataFrame, list[str]]:
	"""Annotate a set of seqlets according to a motif database using TOMTOM.

	This function takes in a set of seqlets and a motif database and assigns
	to each seqlet the most significant motif match. This match is done using
	TOMTOM on just the one-hot encoded sequence for the seqlet and the PWM
	for the motifs. If no motifs have a statistical significance below the
	provided threshold, an index of -1 is returned.

	The computation is independent for each seqlet, so ordering of seqlets
	should not matter, nor should running this function on a subset of seqlets
	versus the full set.


	Parameters
	----------
	X: torch.Tensor, shape=(n_examples, len(alphabet), length)
		A one-hot encoded set of sequences. These are only relevant for
		extracting the sequences spanned by each seqlet.

	seqlets: pandas.DataFrame, shape=(n_seqlets, 3+)
		A BED-formatted set of seqlets to annotate. The first three columns
		must correspond to the example index in `X`, the start, and the end
		(both base-0). The annotations can have additional columns, but those
		are ignored.

	motifs: dict or str
		A dictionary of motifs where the keys are motif names and the values
		are PWMs with shape (len(alphabet), len(motif)), or a string to a
		MEME-formatted file.

	n_nearest: int, optional
		The number of motif matches to return for each seqlet, starting with
		the most significant hit. Default is 1.

	n_jobs: int, optional
		The number of threads to run TOMTOM on in parallel. -1 means use all
		available threads. Default is -1.

	**kwargs: arguments, optional
		Any other arguments to pass into the TOMTOM algorithm.


	Returns
	-------
	idxs: torch.Tensor, shape=(len(seqlets), n_nearest)
		The index of the most significant motif hit(s) for each seqlet.

	pvals: torch.Tensor, shape=(len(seqlets), n_nearest)
		The p-value of the returned significant hits.
	"""

	if isinstance(motifs, str):
		motifs = read_meme(motifs)

	motif_pwms = list(motifs.values())

	X_seqlets = []
	for example_idx, start, end in seqlets.iloc[:, :3].values:
		X_seqlets.append(X[example_idx, :, start:end])

	p_values, _, _, _, _, idxs = tomtom(X_seqlets, motif_pwms, 
		n_nearest=n_nearest, n_jobs=n_jobs, **kwargs)

	return torch.from_numpy(idxs).type(torch.int32), torch.from_numpy(p_values)


def count_annotations(
	X: torch.Tensor | pandas.DataFrame,
	dtype: torch.dtype = torch.uint8,
	shape: tuple[int, int] | None = None,
	dim: int | None = None,
) -> torch.Tensor:
	"""A method for counting the annotations for each example.

	This function takes in a tensor of (example_idx, annotation_idx) pairs and
	returns a tensor of counts where each row is an example, each column is an
	annotation, and the value is the number of times that annotation appears
	in that example. The dimensions of the returned matrix will then be the
	maximum example_idx and motif_idx value even if intermediary values are
	not observed.


	Parameters
	----------
	X: torch.Tensor, shape=(n_annotations, 2) or tuple of two vectors
		A tensor of annotations where the first column is the example_idx and
		the second column is the annotation_idx. Both should be integers. 

	dtype: torch.dtype, optional
		The dtype of the returned matrix. Default is torch.uint8. Note that
		counts above 255 will silently overflow under the default dtype; pass
		a wider dtype (e.g. `torch.int32`) if any (example, annotation) pair
		may occur more than 255 times.

	shape: tuple of (int, int) or None, optional
		A user-defined shape of the returned count matrix. Use this if you are
		not sure whether all examples or annotations are observed in `X` but you
		want to have a constant shape. If None, derive shape from the maximum
		values in `X`. Default is None.

	dim: None, 0, or 1, optional
		Whether to aggregate the counts along one of the axes. If set to None,
		the full count matrix will be returned. If set to 0 the first dimension
		is summed over, returning the number of occurrences of each annotation.
		If set to 1, the second dimension is summed over, returning the number
		of annotations per example. Default is None. 


	Returns
	-------
	y: torch.Tensor, shape=(max(example_idx), max(motif_idx))
		A sparse tensor where each row is an example and each column is an
		annotation and the values within are the number of times each annotation
		appears in each example.
	"""

	if isinstance(X, (list, tuple)):
		x_ = []
		for x in X:
			if isinstance(x, pandas.Series):
				x = x.values

			if isinstance(x, numpy.ndarray):
				x = torch.from_numpy(x)

			x_.append(x)

		X = torch.vstack(x_).T

	_validate_input(X, 'X', shape=(-1, 2), min_value=0)

	X = X.type(torch.int64)
	X_ones = torch.ones(len(X), dtype=dtype)

	n_examples, n_annotations = X.max(dim=0).values + 1

	if shape is not None:
		if n_examples > shape[0] or n_annotations > shape[1]:
			raise RuntimeError("Observed maximum indices {} but".format(
				(n_examples, n_annotations)) + " provided shape {}".format(
				shape))
		
		n_examples, n_annotations = shape


	if dim == 0:
		y = torch.zeros(n_annotations, dtype=dtype)
		y.scatter_add_(0, X[:, 1], X_ones)
	
	elif dim == 1:
		y = torch.zeros(n_examples, dtype=dtype)
		y.scatter_add_(0, X[:, 0], X_ones)
	
	else:
		X_idxs = X[:, 0] * n_annotations + X[:, 1]
		
		y = torch.zeros(n_examples * n_annotations, dtype=dtype)
		y.scatter_add_(0, X_idxs, X_ones)
		y = y.reshape(n_examples, n_annotations)
	
	return y


def pairwise_annotations(
	X: torch.Tensor | pandas.DataFrame,
	dtype: torch.dtype = torch.int64,
	unique: bool = True,
	symmetric: bool = True,
	shape: int | None = None,
) -> torch.Tensor:
	"""Returns the number of times pairs of annotations occur in an example.

	This function takes in a tensor of (example_idx, annotation_idx) pairs and
	returns a tensor of counts where each row is an annotation_idx and each
	column is also an annotation_idx, and the values within are the number of
	times that the each pair of annotations appears in the same example.

	The returned matrix will be a symmetric matrix that has a total number of
	counts equal to twice the number of provided examples.


	Parameters
	----------
	X: torch.Tensor, shape=(n_annotations, 2)
		A tensor of annotations where the first column is the example_idx and
		the second column is the annotation_idx. Both should be integers. 

	dtype: torch.dtype, optional
		The dtype of the returned matrix. Default is torch.int64.

	unique: bool, optional
		Whether to count only unique pairs within each example or each instance.
		For example, if set to True, an example with 3 instances of KLF4 would
		contribute only a single KLF4-KLF4 interaction to the matrix, whereas
		if set to False, would contribute many. Default is True.

	symmetric: bool, optional
		Whether to return a symmetric matrix or one where the row is the first
		element in `X` and the column is the subsequent element in `X` (i.e.
		ordering follows the order rows appear in `X`, not the spatial layout
		of the annotations). If symmetric, the diagonal is NOT double counted.
		Default is True.

	shape: int or None, optional
		The number of rows and columns to use in the matrix. If None, infer the
		number from `X`. Default is None.


	Returns
	-------
	y: torch.Tensor, shape=(max(motif_idx), max(motif_idx))
		A tensor where each row and column corresponds to an annotation index
		and the values within are the number of times that pair of annotations
		appears in the same example.
	"""

	if isinstance(X, (list, tuple)):
		x_ = []
		for x in X:
			if isinstance(x, pandas.Series):
				x = x.values

			if isinstance(x, numpy.ndarray):
				x = torch.from_numpy(x)

			x_.append(x)

		X = torch.vstack(x_).T

	_validate_input(X, 'X', shape=(-1, 2), min_value=0)

	n_examples, n_annotations = X.max(dim=0).values + 1

	if shape is not None:
		if n_annotations > shape:
			raise RuntimeError("Observed maximum indices {} but".format(
				(n_examples, n_annotations)) + " provided shape {}".format(
				shape))
		
		n_annotations = shape

	example_annotations = [[] for i in range(n_examples)]
	for example_idx, annotation_idx in X:
		example_annotations[example_idx].append(annotation_idx)

	y = torch.zeros(n_annotations, n_annotations, dtype=dtype).numpy()
	for annotations in example_annotations:
		if unique:
			annotations = numpy.unique(annotations)
		
		for i, idx0 in enumerate(annotations[:-1]):
			for j, idx1 in enumerate(annotations[i+1:]):
				y[idx0, idx1] += 1

				if symmetric and idx0 != idx1:
					y[idx1, idx0] += 1

	return torch.from_numpy(y)


def _check_spacing(d, start0, end0, start1, end1):
	"""Raise if a pair of annotations overlaps, giving a negative distance.

	The distance between two annotations is the gap between the end of the
	left one and the start of the right one. When they overlap this quantity
	is negative, which would index from the far end of the distance axis and
	be silently indistinguishable from a pair `max_distance + d` apart.
	"""

	if d < 0:
		raise ValueError("pairwise_annotations_spacing requires that "
			"annotations within an example do not overlap, because "
			"overlapping spans produce a negative distance; got the pair "
			f"[{start0}, {end0}) and [{start1}, {end1}), which overlap by "
			f"{-d}. Drop or merge overlapping annotations before calling.")


def pairwise_annotations_spacing(
	X: torch.Tensor | pandas.DataFrame,
	max_distance: int = 100,
	dtype: torch.dtype = torch.uint8,
	symmetric: bool = True,
	shape: int | None = None,
) -> torch.Tensor:
	"""Finds the number of times each annotation pair occurs at each distance.

	This function takes in a tensor of (example_idx, annotation_idx, start, end) 
	tuples and returns a tensor of counts where the first two dimensions are 
	annotation indexes and the third dimension is the spacing between the pair. 
	The values within the tensor are the count of the number of times that pair 
	of annotations is found with that spacing. Importantly, distance is 
	calculated between the end of the motif on the left and the start of the 
	motif on the right. Put another way, it is the number of nucleotides between 
	each motif and so is invariant to their lengths.

	The returned tensor will be extremely sparse, so be careful about setting
	the maximum distance to a value that is very high. A sparse tensor could
	be used here, but given how cheap memory is getting this function is meant
	to be a quick solution that handles most cases.


	Parameters
	----------
	X: torch.Tensor, shape=(n_annotations, 4)
		A tensor of annotations where the columns should be the example_idx,
		annotation_idx, start of the annotation, and end of the annotation.
		The end of the annotation should not be inclusive, so two adjacent
		annotations with zero spacing between them should have the same integer
		index for the end of one motif and the start of the next. Annotations
		within an example must not overlap, because the distance between an
		overlapping pair is negative and has no bin in the returned tensor;
		an overlapping pair raises a `ValueError`.

	max_distance: int, optional
		The maximum distance between two annotations in the same example to
		consider. Default is 100.

	dtype: torch.dtype, optional
		The dtype of the returned matrix. Default is torch.uint8. Note that
		counts above 255 will silently overflow under the default dtype; pass
		a wider dtype (e.g. `torch.int32`) if any (motif_a, motif_b, distance)
		bin may occur more than 255 times.

	symmetric: bool, optional
		Whether to return a symmetric matrix or one where the row is the
		spatially leftmost annotation of the pair and the column is the
		spatially rightmost (i.e. ordering is determined by `start` position
		within each example, NOT by the order rows appear in `X`; this
		differs from `pairwise_annotations`). If symmetric, the diagonal is
		NOT double counted. Default is True.

	shape: int or None, optional
		The number of rows and columns to use in the matrix. If None, infer the
		number from `X`. Default is None.


	Returns
	-------
	y: torch.Tensor, shape=(max(motif_idx), max(motif_idx), max_distance)
		A tensor where the first two dimensions are annotation indices and the
		third dimension is the spacing between the pair, and the values within
		are the count of the number of times that pair of annotations is found
		with that spacing.
	"""

	if isinstance(X, (list, tuple)):
		x_ = []
		for x in X:
			if isinstance(x, pandas.DataFrame):
				x = torch.from_numpy(x.values)

			elif isinstance(x, numpy.ndarray):
				x = torch.from_numpy(x)

			if x.ndim == 1:
				x = x.unsqueeze(1)

			x_.append(x)

		X = torch.cat(x_, axis=1)[:, [0, 3, 1, 2]]

	elif isinstance(X, pandas.DataFrame):
		X = torch.from_numpy(X.values)

	if X.shape[0] == 0:
		raise ValueError("pairwise_annotations_spacing requires at least one "
			"annotation; got X with shape[0] == 0.")

	_validate_input(X, 'X', shape=(-1, 4), min_value=0)

	n_examples, n_annotations = X.max(dim=0).values[:2] + 1

	if shape is not None:
		if n_annotations > shape:
			raise RuntimeError("Observed maximum indices {} but".format(
				(n_examples, n_annotations)) + " provided shape {}".format(
				shape))
		
		n_annotations = shape


	example_annotations = [[] for i in range(n_examples)]

	y = torch.zeros(n_annotations, n_annotations, max_distance, 
		dtype=dtype).numpy()

	for example_idx, annotation_idx, start, end in X:
		example_annotations[example_idx].append((annotation_idx, start, end))

	for annotations in example_annotations:
		for i, (idx0, start0, end0) in enumerate(annotations[:-1]):
			for j, (idx1, start1, end1) in enumerate(annotations[i+1:]):
				if start0 < start1:
					d = start1 - end0
					if d >= max_distance:
						continue

					_check_spacing(d, start0, end0, start1, end1)

					y[idx0, idx1, d] += 1
					if symmetric and idx0 != idx1:
						y[idx1, idx0, d] += 1

				else:
					d = start0 - end1
					if d >= max_distance:
						continue

					_check_spacing(d, start1, end1, start0, end0)

					y[idx1, idx0, d] += 1
					if symmetric and idx0 != idx1:
						y[idx0, idx1, d] += 1

	return torch.from_numpy(y)
