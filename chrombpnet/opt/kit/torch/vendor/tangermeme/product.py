# product.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import torch

from tqdm import tqdm

from ._compat import _preserve_model_state, _resolve_device


def _apply(func, model, X, args, batch_size, device, verbose,
	additional_func_kwargs, **kwargs):
	"""An internal function for applying a function to a batch of data."""

	X = torch.stack(X)
	args = [torch.stack(a) for a in args]

	y = func(model, X, args=args, batch_size=batch_size, device=device,
		verbose=False, **additional_func_kwargs, **kwargs)

	return y


def apply_pairwise(
	func: Callable[..., Any],
	model: torch.nn.Module,
	X: torch.Tensor,
	args: Sequence[torch.Tensor] | None = None,
	batch_size: int = 32,
	device: str | torch.device | None = None,
	additional_func_kwargs: dict | None = None,
	verbose: bool = False,
	**kwargs: Any,
) -> torch.Tensor | list[torch.Tensor]:
	"""Apply a function on the cartesian product between X and args.

	This function will take the provided function and apply it in a batched
	manner across the cartesian product of `X` and `args`, with the assumption
	that each arguments in args is the same length. Basically, this function
	should be used when there are two axes on which to apply the function --
	the first being sequence, and the second being anything else -- and each
	of the arguments in `args` describe that second axis. This will return one
	or more tensors whose first axes are (len(X), len(args[0])). This is in
	contrast to `apply_product`, which returns tensors whose first axes would
	be `(len(X), len(args[0]), len(args[1])...). 

	As a more specific example, DragoNNFruit can make predictions for sequences 
	in  each cell in a single-cell experiment. Each cell is represented by a
	vector, which is one of the arguments, but also by a single number that is
	the read depth of the cell. Using `apply_product` would be incorrect in
	this setting because we are not interested in the cross between cell state
	and read depth. Rather, we want the cell state and read depth information
	to be paired. 


	Parameters
	----------
	func: function
		A function, likely implemented in tangermeme, to apply in a batched
		manner across the product of examples.

	model: torch.nn.Module
		The PyTorch model to use to make predictions.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to make predictions for.

	args: tuple or list
		A set of additional arguments, each of which is one input to the model.
		Unlike `apply_product`, every element in `args` must have the same
		length and is paired index-wise (so `args[0][i]` is grouped with
		`args[1][i]`); the cross is taken with `X`, not within `args`. The
		resulting output has shape `(len(X), len(args[0]), ...)`.

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. The
		model's original device and training mode are restored after the call.
		Default is None.

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
		Whether to display a progress bar as examples are processed. Default
		is False.

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y: torch.Tensor or list/tuple of torch.Tensors
		The output from the model for each input example. The precise format
		is determined by the model. If the model outputs a single tensor,
		y is a single tensor concatenated across all batches. If the model
		outputs multiple tensors, y is a list of tensors which are each
		concatenated across all batches.
	"""

	device = _resolve_device(device)
	additional_func_kwargs = dict(additional_func_kwargs or {})
	# Pass device as a string to downstream `func` to remain back-compatible
	# with callables that haven't yet been updated to accept torch.device.
	device_arg = str(device)

	if len(X) == 0:
		raise ValueError("apply_pairwise requires at least one example; "
			"got X with shape[0] == 0.")

	if args is not None and any(len(a) == 0 for a in args):
		raise ValueError("apply_pairwise requires every element in `args` to "
			"be non-empty; got at least one zero-length entry.")

	if args is not None and len(args) > 1:
		lengths = [len(a) for a in args]
		if len(set(lengths)) != 1:
			raise ValueError("apply_pairwise requires every element in `args` "
				"to have the same length (they are paired index-wise). Got "
				f"lengths {lengths}; use `apply_product` for the cartesian "
				"product instead.")

	with _preserve_model_state(model, device):
		X_, y, args_ = [], [], [[] for _ in args]
		for x in tqdm(itertools.product(X, zip(*args)), disable=not verbose):
			X_.append(x[0])

			for i, arg in enumerate(x[1]):
				args_[i].append(arg)

			if len(X_) == batch_size:
				y_ = _apply(func, model, X_, args=args_, batch_size=batch_size,
					device=device_arg, verbose=verbose,
					additional_func_kwargs=additional_func_kwargs, **kwargs)
				y.append(y_)

				X_, args_ = [], [[] for _ in args]
		else:
			if len(X_) > 0:
				y_ = _apply(func, model, X_, args=args_, batch_size=batch_size,
					device=device_arg, verbose=verbose,
					additional_func_kwargs=additional_func_kwargs, **kwargs)
				y.append(y_)

	Xal = [len(X), len(args[0])]

	# If there is only a single output, just concatenate the tensors
	if isinstance(y[0], torch.Tensor):
		yl = y[0].shape[1:]
		y = torch.cat(y).reshape(*Xal, *yl)
	else:
		_y = []

		# If either the function or the model have multiple outputs, but the
		# other has a single output, then concatenate tensors across the
		# outputs appropriately.
		if isinstance(y[0][0], torch.Tensor):
			for y_ in list(zip(*y)):
				yl = y_[0].shape[1:]
				_y.append(torch.cat(y_).reshape(*Xal, *yl))

		# If both the function and the model have multiple outputs then you
		# have to go one layer deeper when concatenating the tensors.
		else:
			for y_task in list(zip(*y)):
				_y.append([])

				for y_ in list(zip(*y_task)):
					yl = y_[0].shape[1:]
					_y[-1].append(torch.cat(y_).reshape(*Xal, *yl))

		y = _y

	return y


def apply_product(
	func: Callable[..., Any],
	model: torch.nn.Module,
	X: torch.Tensor,
	args: Sequence[torch.Tensor],
	batch_size: int = 32,
	device: str | torch.device | None = None,
	additional_func_kwargs: dict | None = None,
	verbose: bool = False,
	**kwargs: Any,
) -> torch.Tensor | list[torch.Tensor]:
	"""Apply a function on the cartesian product between X and each args.

	This function will take the provided function and apply it in a batched
	manner across the cartesian product of `X` and each of the arguments 
	provided in `args`. Because this is a cartesian product, the number of
	examples that need to be processed will quickly grow with respect to the
	number of arguments being passed in. Each of the tensors in `args` must
	be one input to `model`, in the order that they are specified by the
	forward function. 

	This function can accept in any other function -- be it predictions,
	attributions, or marginalizations. If the provided function itself has
	parameters that need to be specified, you can provide them directly to
	this function in the order that they appear in the provided function.


	Parameters
	----------
	func: function
		A function, likely implemented in tangermeme, to apply in a batched
		manner across the product of examples.

	model: torch.nn.Module
		The PyTorch model to use to make predictions.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A one-hot encoded set of sequences to make predictions for.

	args: tuple or list
		A set of additional arguments to pass into the model. Each element in
		`args` should be one tensor that is input to the model. The elements do
		not need to be the same size as each other as a product will be
		constructed over all of them, as well as with `X`. If you only want
		to use one value for an argument across all function applications,
		pass it in as a length-1 tensor.

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. The
		model's original device and training mode are restored after the call.
		Default is None.

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
		Whether to display a progress bar as examples are processed. Default
		is False.

	kwargs: optional
		Additional named arguments that will get passed into the function when
		it is called. Default is no arguments are passed in.


	Returns
	-------
	y: torch.Tensor or list/tuple of torch.Tensors
		The output from the model for each input example. The precise format
		is determined by the model. If the model outputs a single tensor,
		y is a single tensor concatenated across all batches. If the model
		outputs multiple tensors, y is a list of tensors which are each
		concatenated across all batches.
	"""

	device = _resolve_device(device)
	additional_func_kwargs = dict(additional_func_kwargs or {})
	# Pass device as a string to downstream `func` to remain back-compatible
	# with callables that haven't yet been updated to accept torch.device.
	device_arg = str(device)

	if len(X) == 0:
		raise ValueError("apply_product requires at least one example; "
			"got X with shape[0] == 0.")

	if args is not None and any(len(a) == 0 for a in args):
		raise ValueError("apply_product requires every element in `args` to "
			"be non-empty; got at least one zero-length entry.")

	with _preserve_model_state(model, device):
		X_, y, args_ = [], [], [[] for _ in args]
		for x in tqdm(itertools.product(X, *args), disable=not verbose):
			X_.append(x[0])

			for i, arg in enumerate(x[1:]):
				args_[i].append(arg)

			if len(X_) == batch_size:
				y_ = _apply(func, model, X_, args=args_, batch_size=batch_size,
					device=device_arg, verbose=verbose,
					additional_func_kwargs=additional_func_kwargs, **kwargs)
				y.append(y_)

				X_, args_ = [], [[] for _ in args]
		else:
			if len(X_) > 0:
				y_ = _apply(func, model, X_, args=args_, batch_size=batch_size,
					device=device_arg, verbose=verbose,
					additional_func_kwargs=additional_func_kwargs, **kwargs)
				y.append(y_)

	Xal = [len(X)] + [len(a) for a in args]

	# If there is only a single output, just concatenate the tensors
	if isinstance(y[0], torch.Tensor):
		yl = y[0].shape[1:]
		y = torch.cat(y).reshape(*Xal, *yl)
	else:
		_y = []

		# If either the function or the model have multiple outputs, but the
		# other has a single output, then concatenate tensors across the
		# outputs appropriately.
		if isinstance(y[0][0], torch.Tensor):
			for y_ in list(zip(*y)):
				yl = y_[0].shape[1:]
				_y.append(torch.cat(y_).reshape(*Xal, *yl))

		# If both the function and the model have multiple outputs then you
		# have to go one layer deeper when concatenating the tensors.
		else:
			for y_task in list(zip(*y)):
				_y.append([])

				for y_ in list(zip(*y_task)):
					yl = y_[0].shape[1:]
					_y[-1].append(torch.cat(y_).reshape(*Xal, *yl))

		y = _y

	return y
