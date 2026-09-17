# saturation_mutagenesis.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any
from typing import NamedTuple

import torch

from tqdm import trange

from .predict import predict
from .utils import TangermemeWarning


class SaturationMutagenesisRawResult(NamedTuple):
	"""Return type of `saturation_mutagenesis` when `raw_outputs=True`.

	Holds the reference predictions (`y0`, model output on the original
	sequences) alongside `y_hat`, the model output for each
	edit-distance-one perturbation. Positional unpacking
	(`y0, y_hat = saturation_mutagenesis(..., raw_outputs=True)`)
	continues to work.
	"""

	y0: torch.Tensor | list[torch.Tensor]
	y_hat: torch.Tensor | list[torch.Tensor]


def _attribution_score(y0, y_hat, target):
	"""An internal function for calculating the ISM attributions.

	This function, which is meant to be used for ISM, will take in the
	predictions before and after substitutions and a target -- which can be
	None -- and return the position-normalized differences. Specifically,
	for each example, the differences in prediction will be normalized by
	subtracting out the per-position average, and then averaged across
	all tasks if target is None.


	Parameters
	----------
	y0: torch.Tensor, shape=(-1, n_targets)
		Model predictions for each example on the original predictions.

	y_hat: torch.Tensor, shape=(-1, len(alphabet), length, n_targets)
		Model predictions for each example for each substitution.

	target: int or slice or None
		If the user wants to subset to only some targets when calculating the
		average attribution across targets. If None, use all.
	"""

	attr = y_hat[:, :, :, target] - y0[:, None, None, target]
	attr -= torch.mean(attr, dim=1, keepdims=True)

	if len(attr.shape) > 3:
		attr = torch.mean(attr, dim=tuple(range(3, len(attr.shape))))
	return attr



def saturation_mutagenesis(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: tuple | None = None,
	start: int = 0,
	end: int = -1,
	batch_size: int = 32,
	target: int | slice | None = None,
	hypothetical: bool = False,
	raw_outputs: bool = False,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	verbose: bool = False,
	func: Callable[..., Any] | None = None,
) -> torch.Tensor | SaturationMutagenesisRawResult:
	"""Performs in-silico saturation mutagenesis on a set of sequences.
	
	This function will perform in-silico saturation mutagenesis on a set of 
	sequences and return the predictions on the original sequences and each
	of the sequences with an edit distance of one on them.
	
	By default, this function will aggregate these predictions into an
	attribution value. For each single-character substitution, the change in
	prediction (perturbed minus original) is computed and then mean-subtracted
	across the alphabet axis so the values at each position are centered on zero;
	when `target=None` these are additionally averaged across all of the model's
	outputs. The values are not Euclidean distances and are not Z-score
	normalized -- preserving the raw magnitude lets you compare the importance of
	different regions head-to-head. Unless `hypothetical=True`, the result is then
	projected onto the observed bases by multiplying by the one-hot `X`. This
	aggregation assumes that the model returns only a single tensor. This tensor
	can have multiple outputs, e.g., be of shape (batch_size, n_targets) where
	n_targets > 1, but the model cannot return multiple tensors.
	
	If you simply want the predictions before and after the substitutions
	without the method turning those into attributions because, perhaps, you
	want to define your own aggregation method, you can use `raw_outputs=True`.
	
	
	Parameters
	----------
	model: torch.nn.Module
		The PyTorch model to use to make predictions.
	
	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate attribution values
		for. 
	
	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.
	
	start: int, optional
		The start of where to begin making perturbations to the sequence.
		Default is 0.
	
	end: int, optional
		The end of where to make perturbations to the sequence. Positive
		values follow standard Python slice semantics (non-inclusive).
		Negative values are remapped via `end = length + 1 + end` so that
		`end=-1` maps to `length` and the *last* nucleotide is included.
		Note this differs from Python's `X[start:-1]` (which would drop the
		last element). Default is -1, meaning the entire sequence.
	
	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.
	
	target: int or slice or None, optional
		Whether to focus on a single output/slice of outputs from the model
		when calculating attributions rather than the entire set of outputs.
		If None, use all targets when calculating attributions. Default is None.
	
	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position or only for the character that is actually in the sequence.
		Only matters when `raw_outputs=False`. Default is False.
	
	raw_outputs: bool, optional
		Whether to return the raw outputs from the method -- in this case,
		the predictions from the reference sequence and from each of the
		perturbations -- or the processed attribution values. Default is False.
	
	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Default is None.
	
	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. Default
		is None.
	
	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.

	func: function or None, optional
		A function to apply to a batch of predictions after they have been made,
		forwarded to `predict`. It is applied identically to the reference
		predictions and to every perturbation, so attributions are computed on
		the post-processed values. Use it to, e.g., select a single output head
		or apply a final non-linearity. If None, do nothing. Default is None.


	Returns
	-------
	attr: torch.Tensor
		Processed attribution values. For each example and each position, the
		difference between the model's prediction on the perturbed sequence and
		on the original sequence is computed, then mean-subtracted across the
		alphabet axis so the values at each position are centered on zero.
	
	-- or, if raw_outputs=True --
	
	y0: torch.Tensor or list/tuple of torch.Tensors
		The outputs from the model for the reference sequences.
	
	y_hat: torch.Tensor or list/tuple of torch.Tensors
		The outputs from the model for each of the perturbed sequences.
	"""

	if X.is_floating_point() and not torch.equal(X, X.round()):
		warnings.warn("X contains non-integer values which will be truncated "
			"toward zero by the int8 cast; values with magnitude < 1 become 0. "
			"Pass a hard one-hot encoding to avoid silently zeroing positions.",
			TangermemeWarning)

	X = X.type(torch.int8).cpu()
	y0 = predict(model, X, args=args, func=func, dtype=dtype, device=device)

	if not raw_outputs and not isinstance(y0, torch.Tensor):
		raise ValueError("raw_outputs=True is required for models that return "
			"multiple output tensors; the attribution aggregation assumes a "
			"single output tensor.")

	length = X.shape[-1]
	if end < 0:
		end = length + 1 + end

	if start < 0 or end > length or start >= end:
		raise ValueError("start and end must satisfy "
			"0 <= start < end <= length; got start={}, end={} for length "
			"{}.".format(start, end, length))

	y_hat = []
	for i in trange(X.shape[0], disable=not verbose):
		# Build only the true single-base edits. One substitution per position
		# would re-apply the base already there (an identity edit whose
		# prediction equals y0); it is skipped here and filled from y0 below,
		# saving up to 25% of the forward passes. A column that is all-zero (an
		# `N`) or multi-hot is not a clean one-hot of any character, so it has
		# no identity row and all four of its edits are kept. The kept rows are
		# laid out in [character, position] order to match the reconstruction.
		ref = X[i, :, start:end]
		identity = (ref == 1) & (ref.sum(dim=0, keepdim=True) == 1)
		edits = ~identity.reshape(-1)

		edit_chars, edit_positions = torch.where(~identity)
		edit_positions = edit_positions + start
		n_edits = edit_chars.shape[0]

		X_ = X[i].repeat(n_edits, 1, 1)
		rows = torch.arange(n_edits)
		X_[rows, :, edit_positions] = 0
		X_[rows, edit_chars, edit_positions] = 1

		if args is not None:
			args_ = tuple(a[i].repeat(n_edits, *(1 for _ in a[i].shape))
				for a in args)
		else:
			args_ = None

		y_edits = predict(model, X_, args=args_, func=func,
			batch_size=batch_size, dtype=dtype, device=device)

		# Scatter the edit predictions back into the full [character, position]
		# grid, filling the identity slots with the reference prediction y0.
		if isinstance(y_edits, torch.Tensor):
			y_hat_ = y0[i].unsqueeze(0).repeat(edits.shape[0],
				*(1 for _ in y0[i].shape))
			y_hat_[edits] = y_edits
		else:
			y_hat_ = [y0_[i].unsqueeze(0).repeat(edits.shape[0],
				*(1 for _ in y0_[i].shape)) for y0_ in y0]
			for y_hat_h, y_edit in zip(y_hat_, y_edits):
				y_hat_h[edits] = y_edit

		y_hat.append(y_hat_)

	if isinstance(y_hat[0], torch.Tensor):
		y_hat = torch.stack(y_hat).reshape(X.shape[0], X.shape[1], end-start, 
			*y_hat_.shape[1:])
	else:
		y_hat = [
			torch.stack(y_).reshape(X.shape[0], X.shape[1], end-start,
				*y_[0].shape[1:]) for y_ in zip(*y_hat)
		]
	
	if not raw_outputs:
		attr = _attribution_score(y0, y_hat, target)
		return X[:, :, start:end] * attr if not hypothetical else attr
	return SaturationMutagenesisRawResult(y0=y0, y_hat=y_hat)
