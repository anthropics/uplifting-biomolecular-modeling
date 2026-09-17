# marginalize.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy
import torch

from .utils import _validate_input
from .utils import one_hot_encode

from .ersatz import substitute
from .predict import predict
from .results import PerturbationResult, PerturbationAnnotationsResult


def marginalize(
	model: torch.nn.Module,
	X: torch.Tensor,
	motif: torch.Tensor | str,
	start: int | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	func: Callable[..., Any] = predict,
	additional_func_kwargs: dict | None = None,
	**kwargs: Any,
) -> PerturbationResult:
	"""Apply a function before and after substituting a motif into sequences.

	A marginalization experiment is one where a function is applied before
	and after substituting something into a set of sequences. It is named as
	such because the sequences are meant to be background sequences and the
	difference in output before and after the substitution represents the
	"marginal" effect of adding that something into the sequences. When you are
	adding a motif to the sequence, the difference in output can be interpreted 
	as the effect that motif has on the function in isolation.

	By default, `marginalize` will apply the `predict` function to `X` before
	and after substituting in a one-hot encoded version of `motif`. However,
	one can pass in any function, including `deep_lift_shap` or even
	`saturation_mutagenesis`. These functions may have additional arguments
	and those can be passed into `marginalize` as-is and will be passed along
	to the function. If any arguments would have had the same name as those
	used by this function, you can use the `additional_func_kwargs` input to
	ensure those values get to the function.

	Naturally, most models being used with tangermeme will be non-linear and
	so the marginal effect of each motif is only somewhat useful because motifs
	do not occur in isolation in the genome. Other functions, such as `space`,
	can be invaluable in seeing how motifs interact with each other. However,
	looking at the marginal effect of each motif can still be invaluable
	because it gives you a sense for what motifs yield an effect at all and
	roughly how strong that effect is.
	

	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have a motif inserted into.

	motif: torch.tensor or str
		Either a one-hot encoded short motif with shape
		`(-1, len(alphabet), motif_length)` to insert into the set of
		sequences, or a string that will be one-hot encoded internally using
		`alphabet` (and the `N` ignore convention from `ersatz.substitute`).

	start: int or None, optional
		The starting position of where to insert the motif. If None, insert the
		motif into the middle of the sequence such that the middle of the motif
		occurs at the middle of the sequence. Default is None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

	func: function, optional
		A function to apply before and after making the substitution. Default 
		is `predict`.

	additional_func_kwargs: dict or None, optional
		Additional named arguments to pass into the function when it is called.
		This is provided as an alternate path to route arguments into the
		function in case they overlap, name-wise, with those in this function,
		or if you want to be absolutely sure that the arguments are making
		their way into the function. The dict is not modified in place. Default
		is None.

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y_before: torch.Tensor or list of torch.Tensors
		The output from the function before inserting the motif in. If the
		output is a single tensor, it will return that. If the model outputs a 
		list of tensors, it will return those.

	y_after: torch.Tensor or list of torch.Tensors
		The output from the function after inserting the motif in. If the
		output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.
	"""

	_validate_input(X, "X", shape=(-1, len(alphabet), -1), ohe=True, allow_N=True)
	additional_func_kwargs = dict(additional_func_kwargs or {})

	if isinstance(motif, torch.Tensor) and motif.device != X.device:
		raise ValueError(
			f"`motif` and `X` must be on the same device; got motif on "
			f"{motif.device} and X on {X.device}.")

	X_perturb = substitute(X, motif, start=start, alphabet=alphabet)
	y_before = func(model, X, **kwargs, **additional_func_kwargs)
	y_after = func(model, X_perturb, **kwargs, **additional_func_kwargs)

	return PerturbationResult(y_before=y_before, y_after=y_after)


def marginalize_annotations(
	model: torch.nn.Module,
	X: torch.Tensor,
	X0: torch.Tensor,
	annotations: torch.Tensor,
	**kwargs: Any,
) -> PerturbationAnnotationsResult:
	"""Perform marginalizations on each annotation individually.

	This function takes in a model, a set of sequences, a set of background
	sequences, and a set of annotations, and returns the marginalization values
	for each annotation. For each annotation, the sequence in `X` is extracted
	and substituted into `X0` with predictions returned for `X0` before and
	after the substitution is performed, similar to the `saturation_mutagenesis`
	function. Each marginalization is done individually.

	This function will extract the sequence in each annotation and perform a
	marginalization on it individually. 


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences corresponding to the annotations.

	X0: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences that motifs will be substituted into.

	annotations: torch.Tensor, shape=(n_annotations, 3)
		A tensor of annotations where the first column is the example_idx, the
		second column is the start position (0-indexed) and the third column is
		the end position (0-indexed, not inclusive).

	kwargs: arguments
		Additional optional arguments to pass into the `marginalize` function.

	
	Returns
	-------
	y_befores: torch.Tensor or list of torch.Tensors
		The application of `func` from the model BEFORE inserting the motif. If 
		the output from the model's forward function is a single tensor, it will 
		return that. If the model outputs a list of tensors, it will return 
		those.

	y_afters: torch.Tensor or list of torch.Tensors
		The application of `func` from the model AFTER inserting the motif. If 
		the output from the model's forward function is a single tensor, it will
		return that. If the model outputs a list of tensors, it will return
		those.
	"""

	if len(annotations) == 0:
		raise ValueError("marginalize_annotations requires at least one "
			"annotation; got an empty annotations tensor.")

	y_befores, y_afters = [], []

	for idx, start, end in annotations:
		seq = X[idx, :, start:end].unsqueeze(0)

		y_before, y_after = marginalize(model, X0, seq, **kwargs)
		y_befores.append(y_before)
		y_afters.append(y_after)

	if isinstance(y_afters[0], torch.Tensor):
		y_befores = torch.stack(y_befores)
		y_afters = torch.stack(y_afters)
	else:
		y_befores = [torch.stack([x[i] for x in y_befores]) for i in range(len(
			y_befores))]
		y_afters = [torch.stack([x[i] for x in y_afters]) for i in range(len(
			y_afters))]

	return PerturbationAnnotationsResult(y_befores=y_befores, y_afters=y_afters)
