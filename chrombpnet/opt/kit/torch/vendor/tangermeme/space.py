# space.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import numpy
import torch

from .utils import _validate_input
from .utils import _cast_as_tensor
from .utils import one_hot_encode

from .ersatz import multisubstitute
from .predict import predict

from tqdm import tqdm


class SpaceResult(NamedTuple):
	"""Return type of `space`. Positional unpacking
	`y_before, y_afters = space(...)` still works.

	`y_afters` stacks one entry per spacing combination along axis 1
	(after the example axis).
	"""

	y_before: torch.Tensor | list[torch.Tensor]
	y_afters: torch.Tensor | list[torch.Tensor]


def space(
	model: torch.nn.Module,
	X: torch.Tensor,
	motifs: list[torch.Tensor | str],
	spacing: torch.Tensor | numpy.ndarray | list,
	start: int | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	func: Callable[..., Any] = predict,
	additional_func_kwargs: dict | None = None,
	verbose: bool = False,
	**kwargs: Any,
) -> SpaceResult:
	"""Runs a single spacing experiment and returns predictions.

	Given a predictive model, a set of motifs to insert and the spacings
	between them, and a set of background sequences, return the predictions 
	from the model when using the background sequences and after inserting 
	the motifs into the sequences.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have motifs inserted into.

	motifs: list of torch.tensor, shape=(-1, len(alphabet), motif_length)
		A list of strings or of one-hot encoded versions of short motifs to
		substitute into the set of sequences.

	spacing: torch.Tensor, shape=(-1, len(motifs)-1)
		A tensor specifying all spacings between motifs to consider. Each row
		in this tensor is a different combination of spacings between motifs
		and each column is the spacing between an adjacent pair of motifs.
		Specifically, the 1st column corresponds to the spacing between the
		first and second motif, the 2nd column corresponds to the spacing
		between the second and third motif, etc. Must be 2-D; scalar/1-D
		`spacing` is rejected here (unlike `multisubstitute`, which accepts a
		single int and broadcasts it).

	start: int or None, optional
		The starting position of where to insert the first motif. If None, the
		full motif arrangement is centered such that its midpoint coincides with
		the middle of the sequence. Default is None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

	func: function, optional
		A function to apply before and after making the substitutions. Default 
		is `predict`.

	additional_func_kwargs: dict or None, optional
		Additional named arguments to pass into the function when it is called.
		This is provided as an alternate path to route arguments into the
		function in case they overlap, name-wise, with those in this function,
		or if you want to be absolutely sure that the arguments are making
		their way into the function. The dict is not modified in place. Note
		that overlap between keys of `additional_func_kwargs` and `**kwargs`
		raises `TypeError: multiple values for keyword argument` from
		Python's call-site dict-unpacking; only collisions with *this*
		function's named parameters are resolved by routing through
		`additional_func_kwargs`. Default is None.

	verbose: bool, optional
		Whether to display a progress bar as spacings are evaluated. Only
		drives the outer spacing-loop progress bar; it is NOT forwarded to
		inner `func` calls. Default is False.

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y_before: torch.Tensor or list of torch.Tensors
		The predictions from the model before inserting the motifs in, with
		shape `(batch, ...)`. If the output from the model's forward function
		is a single tensor, it will return that. If the model outputs a list
		of tensors, it will return those.

	y_afters: torch.Tensor or list of torch.Tensors
		The predictions from the model after inserting the motifs in, stacked
		over spacings as axis 1: shape `(batch, n_spacings, ...)`. If the
		output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.
	"""

	spacing = _validate_input(_cast_as_tensor(spacing, dtype=torch.int32),
		"spacing", shape=(-1, len(motifs)-1))
	X = _validate_input(X, "X", shape=(-1, len(alphabet), -1))

	for i, motif in enumerate(motifs):
		if isinstance(motif, torch.Tensor) and motif.device != X.device:
			raise ValueError(
				f"motifs[{i}] and X must be on the same device; got "
				f"motifs[{i}] on {motif.device} and X on {X.device}.")

	additional_func_kwargs = dict(additional_func_kwargs or {})

	y_before = func(model, X, **kwargs, **additional_func_kwargs)
	y_afters = []

	for _spacing in tqdm(spacing, disable=not verbose):
		_spacing = [s.item() for s in _spacing]
		
		X_perturb = multisubstitute(X, motifs, _spacing, start=start, 
			alphabet=alphabet)

		y_after = func(model, X_perturb, **kwargs, **additional_func_kwargs)
		y_afters.append(y_after)

	if isinstance(y_afters[0], torch.Tensor):
		y_afters = torch.stack(y_afters).transpose(0, 1)
	else:
		y_afters = [torch.stack(y_).transpose(0, 1) for y_ in list(zip(
			*y_afters))]

	return SpaceResult(y_before=y_before, y_afters=y_afters)
