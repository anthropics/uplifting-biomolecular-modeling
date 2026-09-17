# variant_effect.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy
import torch

from .utils import _cast_as_tensor

from .ersatz import insert

from .predict import predict
from .results import PerturbationResult


def substitution_effect(
	model: torch.nn.Module,
	X: torch.Tensor,
	substitutions: torch.Tensor | numpy.ndarray,
	args: tuple | None = None,
	func: Callable[..., Any] = predict,
	additional_func_kwargs: dict | None = None,
	**kwargs: Any,
) -> PerturbationResult:
	"""Apply a function before and after including one or more substitutions.

	This function will calculate the effect that substitutions have on the
	output from a model. Any number of substitutions can be added to each
	sequence and the provided `func` is applied before and after these
	substitutions are included in the sequences. By default, this `func` is the
	prediction function and so the results are the difference in predictions
	before and after substitutions are made, but if `func` is something else, 
	such as `deep_lift_shap`, the results will be the attributions before and
	after substitutions are made. At least one substitution should be provided 
	per sequence for the results to be different.

	The substitutions provided must be individual variants, i.e., each row
	in the tensor corresponds to a single substitution in a single example, but
	one can encode longer variants (e.g., entire motifs or just multiple
	characters) by passing in multiple rows with adjacent positions.

	Note that substitutions are not insertions. A substitution involves changing
	one character to another character. An insertion involves adding a new
	character into a sequence and requires trimming the edges afterward. Indels
	can be represented as substitutions as long as the indel is one insertion
	and also one deletion.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have substitutions included in.

	substitutions: torch.tensor, shape=(-1, 3)
		A set of variants that should be substituted into each sequence. This
		tensor is formatted like a COO-sparse matrix where each row is a single
		variant, the first column is the index in `X`, the second index is the
		position in that example, and the third index is the index into the
		alphabet that should be present at that position (overriding whatever
		is currently there). Note: all rows for an example are applied to that
		one sequence at once, so at most one row may target any given
		`(example, position)` pair; duplicate coordinates would leave a
		multi-hot column and raise a `ValueError` instead. To score several
		alternate alleles at one position, replicate the example so each
		allele gets its own row of `X`.

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	func: function, optional
		A function to apply before and after incorporating the substitution. 
		Default is `predict`.

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

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y_before: torch.Tensor
		The output from `func` before variants are included.

	y_after: torch.Tensor
		The output from `func` after the variants are included.
	"""

	substitutions = _cast_as_tensor(substitutions)

	# The substitutions are applied with two vectorized assignments over an
	# example-shaped X_var, so rows that collide on the same (example,
	# position) would each set their own 1 and leave a multi-hot column.
	coords = substitutions[:, :2]
	if len(torch.unique(coords, dim=0)) != len(coords):
		raise ValueError("substitution_effect requires at most one "
			"substitution per (example_idx, position); the provided "
			"substitutions contain duplicate coordinates, which would "
			"produce a multi-hot column. De-duplicate them, or replicate "
			"the example so each substitution gets its own row of X.")

	additional_func_kwargs = dict(additional_func_kwargs or {})

	X_var = torch.clone(X)
	X_var[substitutions[:, 0], :, substitutions[:, 1]] = 0
	X_var[substitutions[:, 0], substitutions[:, 2], substitutions[:, 1]] = 1

	y_before = func(model, X, args=args, **additional_func_kwargs, **kwargs)
	y_after = func(model, X_var, args=args, **additional_func_kwargs, **kwargs)
	return PerturbationResult(y_before=y_before, y_after=y_after)


def deletion_effect(
	model: torch.nn.Module,
	X: torch.Tensor,
	deletions: torch.Tensor | numpy.ndarray,
	left: bool = False,
	args: tuple | None = None,
	func: Callable[..., Any] = predict,
	additional_func_kwargs: dict | None = None,
	**kwargs: Any,
) -> PerturbationResult:
	"""Apply a function before and after deleting characters from a sequence.

	This function will calculate the effect that deletions have on the
	output from a model. Any number of deletions can be specified for each
	sequence and the provided `func` is applied before and after the deletions
	are taken into account. By default, this `func` is the prediction function
	and so the results are the difference in predictions before and after the
	deletions are made, but if `func` is something else, such as
	`deep_lift_shap`, the results will be the attributions before and after the
	deletions are made. At least one deletion should be provided per sequence
	for the results to be different.

	The deletions provided must be individual characters, i.e., each row
	in the tensor corresponds to a single deletion in a single example, but one
	can encode longer deletions (e.g., entire motifs or just multiple
	characters) by passing in multiple rows with adjacent positions.

	Importantly, because models assume a fixed input window but deletions are by
	definition changing the length of the sequence, the provided sequences must
	be of length `model_length + max_deletions_per_sequence`. Basically, if the
	maximum number of deletions in a sequence is equal to 12 and the model
	expects a tensor of length 100 then every sequence provided must be of
	length 112, even if there are fewer than 12 deletions in a particular
	sequence.

	Simply removing all of the specified characters will lead to a set of
	sequences of differing lengths if there are a different number of deletions
	in each sequence. In order to make these sequences all the same length, we
	need to trim from each sequence a number of positions such that in total
	(these additional bases + the number of deletions provided by the user)
	characters removed per example is the same for every example. Because there
	are two ways we can trim positions -- either starting from the left end of 
	the sequence or from the right end of it -- you can use the `left` parameter
	to specify that you should trim positions on the left (`left=True`) or from
	the right (`left=False`, the default) of the sequence. 


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length + max_deletions)
		A one-hot encoded set of sequences to have deletions included in.

	deletions: torch.tensor, shape=(-1, 2)
		A set of deletions indicating characters that should be removed from
		each sequence. Each row should be a single deletion with the first
		column corresponding to the index and the second column corresponding
		to the position within that index. Multiple deletions can occur in each
		example. 

	left: bool, optional
		If False, use the first `n` positions to run through the model before
		making the deletion where `n` is the expected tensor length. If True,
		use the last `n` positions. Basically, whether we trim positions from
		the left or the right when getting sequences of the same length.
		Default is False.

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	func: function, optional
		A function to apply before and after incorporating the deletions.
		Default is `predict`.

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

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y_before: torch.Tensor
		The output from `func` on a trimmed slice of `X` — NOT on the raw
		input. Specifically, `X` is reduced to `model_length` columns by
		dropping `max_deletions_per_example` characters from the left when
		`left=True` (otherwise from the right) so the slice fed to the model
		has the same shape as the post-deletion sequence. Use the same
		`left` value the model was trained on.

	y_after: torch.Tensor
		The output from `func` after the variants are included.
	"""

	deletions = _cast_as_tensor(deletions)

	if X.shape[0] == 0:
		raise ValueError("deletion_effect requires at least one example; "
			"got X with shape[0] == 0.")

	additional_func_kwargs = dict(additional_func_kwargs or {})

	mask = torch.zeros_like(X[:, 0]).type(torch.int32)
	mask[deletions[:, 0], deletions[:, 1]] = 1

	max_deletions = int(mask.sum(dim=-1).max())
	if max_deletions >= X.shape[-1]:
		raise ValueError(
			f"deletion_effect requires X.shape[-1] > max deletions per example "
			f"to leave at least one position for the model; got "
			f"X.shape[-1]={X.shape[-1]} and max deletions per example "
			f"={max_deletions}. Each sequence must be of length "
			f"`model_length + max_deletions_per_sequence`.")

	counts = mask.sum(dim=-1)
	counts = abs(counts - counts.max())

	m = mask if left == True else torch.flip(mask, dims=(-1,))
	flank = torch.cumsum(1 - m, dim=-1) <= counts[:, None]
	mask += (flank if left == True else torch.flip(flank, dims=(-1,)))
	mask = (1 - mask).type(torch.bool)
	mask = mask[:, None].repeat(1, X.shape[1], 1) 

	X_var = X[mask].reshape(X.shape[0], X.shape[1], -1)

	if left == True:
		X = X[:, :, -X_var.shape[-1]:]
	else:
		X = X[:, :, :X_var.shape[-1]]

	y_before = func(model, X, args=args, **additional_func_kwargs, **kwargs)
	y_after = func(model, X_var, args=args, **additional_func_kwargs, **kwargs)
	return PerturbationResult(y_before=y_before, y_after=y_after)


def insertion_effect(
	model: torch.nn.Module,
	X: torch.Tensor,
	insertions: torch.Tensor | numpy.ndarray,
	left: bool = False,
	args: tuple | None = None,
	func: Callable[..., Any] = predict,
	additional_func_kwargs: dict | None = None,
	**kwargs: Any,
) -> PerturbationResult:
	"""Apply a function before and after inserting characters into a sequence.

	This function will calculate the effect that insertions have on the
	output from a model. Any number of insertions can be specified for each
	sequence and the provided `func` is applied before and after the insertions
	are taken into account. By default, this `func` is the prediction function
	and so the results are the difference in predictions before and after the
	insertions are made, but if `func` is something else, such as 
	`deep_lift_shap`, the results will be the attributions before and after the
	insertions are made. At least one insertion should be provided per sequence
	for the results to be different.

	The insertions provided must be individual characters, i.e., each row
	in the tensor corresponds to a single insertion in a single example, but one
	can encode longer insertions (e.g., entire motifs or just multiple
	characters) by passing in multiple rows with adjacent positions.

	Because models assume a fixed input window but insertions are by definition
	changing the length of the sequence, characters must be trimmed off after
	the insertions are made so the post-insertion sequence has the same length
	as the original. Because there are two ways we can trim positions -- either
	starting from the left end of the sequence or from the right end of it --
	you can use the `left` parameter to specify that you should trim positions
	on the left (`left=True`) or from the right (`left=False`, the default) of
	the sequence.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to have insertions included in.

	insertions: torch.tensor, shape=(-1, 3)
		A set of insertions indicating characters that should be added to
		each sequence. Each row should be a single insertion with the first
		column corresponding to the example index, the second column
		corresponding to the position within that example at which the
		character should be inserted, and the third column corresponding to
		the index in the alphabet of the character to insert. Multiple
		insertions can occur in each example.

	left: bool, optional
		If False, use the first `n` positions to run through the model after
		making the insertion where `n` is the expected tensor length. If True,
		use the last `n` positions. Basically, whether we trim positions from
		the left or the right when getting sequences of the same length.
		Default is False.

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	func: function, optional
		A function to apply before and after incorporating the insertions.
		Default is `predict`.

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

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y_before: torch.Tensor
		The output from `func` before variants are included.

	y_after: torch.Tensor
		The output from `func` after the variants are included.
	"""

	insertions = _cast_as_tensor(insertions)

	additional_func_kwargs = dict(additional_func_kwargs or {})
	X_var = []

	for i in range(X.shape[0]):
		insertions_ = insertions[insertions[:, 0] == i]
		insertions_ = insertions_[torch.argsort(insertions_[:, 1], 
			descending=True)]

		x = X[i:i+1]
		for _, j, char in insertions_:
			v = torch.zeros(1, X.shape[1], 1)
			v[:, char] = 1
			x = insert(x, v, start=j)

		if left == True:
			x = x[:, :, -X.shape[-1]:]
		else:
			x = x[:, :, :X.shape[-1]]
		
		X_var.append(x)

	X_var = torch.cat(X_var)
	y_before = func(model, X, args=args, **additional_func_kwargs, **kwargs)
	y_after = func(model, X_var, args=args, **additional_func_kwargs, **kwargs)
	return PerturbationResult(y_before=y_before, y_after=y_after)
