# deep_lift_shap.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy
import torch
import torch.nn.functional as F

from tqdm import trange

from ._compat import _autocast_supported, _resolve_device
from .ersatz import dinucleotide_shuffle
from .results import AttributionReferencesResult
from .utils import _validate_input


def hypothetical_attributions(
	multipliers: tuple[torch.Tensor],
	X: tuple[torch.Tensor],
	references: tuple[torch.Tensor],
) -> tuple[torch.Tensor]:
	"""A function for aggregating contributions into hypothetical attributions.

	When handling categorical data, like one-hot encodings, the gradients
	returned by a method like DeepLIFT/SHAP may need to be modified because
	the choice of one character at a position explicitly means that the other
	characters are not there. So, one needs to account for each character change 
	actually being the addition of one character AND the subtraction of another 
	character. Basically, once you've calculated the multipliers, you need to 
	subtract out the contribution of the nucleotide actually present and then
	add in the contribution of the nucleotide you are becoming.

	Each element in the tensor is considered an independent example.

	As an implementation note: to be compatible with Captum, each input must
	be a tuple of length 1 and the returned value will be a tuple of length 1.
	I know this sounds silly but it's the most convenient implementation choice 
	to make the function compatible across DeepLiftShap implementations.


	Parameters
	----------
	multipliers: tuple of one torch.tensor, shape=(n_baselines, 4, length)
		The multipliers/gradient calculated by a method like DeepLIFT/SHAP.
		These should include values for both the observed characters and the
		unobserved characters at each position

	X: tuple of one torch.tensor, shape=(n_baselines, 4, length)
		The one-hot encoded sequence being explained

	references: tuple of one torch.tensor, shape=(n_baselines, 4, length)
		The one-hot encoded reference sequences, usually a shuffled version
		of the corresponding sequence in X.


	Returns
	-------
	projected_contribs: tuple of one torch.tensor, shape=(1, 4, length)
		The attribution values for each nucleotide in the input.
	"""

	for val in multipliers, X, references:
		if not isinstance(val, tuple) or len(val) != 1:
			raise ValueError("All inputs must be one-element tuples.")

		if not isinstance(val[0], torch.Tensor):
			raise ValueError("The first element of each input must be a "
				"tensor.")

		if val[0].shape != multipliers[0].shape:
			raise ValueError("Shape of all tensors must match.") 


	projected_contribs = torch.zeros_like(references[0], dtype=X[0].dtype, 
		device=X[0].device)
	
	for i in range(X[0].shape[1]):
		hypothetical_input = torch.zeros_like(X[0], dtype=X[0].dtype, 
			device=X[0].device)
		hypothetical_input[:, i] = 1.0
		hypothetical_diffs = hypothetical_input - references[0]
		hypothetical_contribs = hypothetical_diffs * multipliers[0]

		projected_contribs[:, i] = torch.sum(hypothetical_contribs, dim=1)

	return (projected_contribs,)


def _register_hooks(module): 
	if len(module._backward_hooks) > 0:
		return
	if not isinstance(module, tuple(module._NON_LINEAR_OPS.keys())):
		return

	module.handles = []
	module.handles.append(module.register_forward_hook(_f_hook))
	module.handles.append(module.register_forward_pre_hook(_fp_hook))
	module.handles.append(module.register_full_backward_hook(_b_hook))


def _clear_hooks(module):
	if hasattr(module, "handles") and len(module.handles) > 0:
		for handle in module.handles:
			handle.remove()

		del module.handles


def _fp_hook(module, inputs): 
	module.input = inputs[0].clone().detach()


def _f_hook(module, inputs, outputs):
	module.output = outputs.clone().detach()


def _b_hook(module, grad_input, grad_output):
	return module._NON_LINEAR_OPS[type(module)](module, grad_input, 
		grad_output)


def _nonlinear(module, grad_input, grad_output):
	"""An internal function implementing a general-purpose nonlinear correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to general non-linear functions such as
	activations.
	"""

	delta_in_ = torch.sub(*module.input.chunk(2))
	delta_out_ = torch.sub(*module.output.chunk(2))

	delta_in = torch.cat([delta_in_, delta_in_])
	delta_out = torch.cat([delta_out_, delta_out_])

	delta = delta_out / delta_in
	idxs = torch.abs(delta_in) < 1e-6

	return (torch.where(idxs, grad_input[0], grad_output[0] * delta),)


def _softmax(module, grad_input, grad_output):
	"""An internal function implementing a correction for softmax activations.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied specifically to softmax activations without
	needing to remove them and operate on the underlying logits.
	"""

	delta_in_ = torch.sub(*module.input.chunk(2))
	delta_out_ = torch.sub(*module.output.chunk(2))

	delta_in = torch.cat([delta_in_, delta_in_])
	delta_out = torch.cat([delta_out_, delta_out_])

	delta = delta_out / delta_in
	idxs = torch.abs(delta_in) < 1e-6

	grad_input_unnorm = torch.where(idxs, grad_input[0], grad_output[0] * delta)

	n = grad_input[0].numel()
	new_grad_inp = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
	return (new_grad_inp,)


def _maxpool(module, grad_input, grad_output):
	"""An internal function implementing a 1D max-pooling correction.

	This function, copied and slightly modified from Captum, is meant to be
	the `rescale` rule applied to max pooling layers given their nature of
	aggregating values across multiple positions.
	"""

	if isinstance(module, torch.nn.MaxPool1d):
		pool_func, unpool_func = F.max_pool1d, F.max_unpool1d
	elif isinstance(module, torch.nn.MaxPool2d):
		pool_func, unpool_func = F.max_pool2d, F.max_unpool2d
	else:
		raise ValueError("module must be either MaxPool1d or MaxPool2d")


	with torch.no_grad():
		delta_in_ = torch.sub(*module.input.chunk(2))
		delta_in = torch.cat([delta_in_, delta_in_])

		output, output_ref = module.output.chunk(2)
		delta_out_xmax = torch.max(output, output_ref)
		delta_out = torch.cat([delta_out_xmax - output_ref, 
			output - delta_out_xmax])

		_, indices = pool_func(module.input, module.kernel_size, module.stride, 
			module.padding, module.dilation, module.ceil_mode, True)

		unpool_ = unpool_func(grad_output[0] * delta_out, indices, 
			module.kernel_size, module.stride, module.padding, 
			list(module.input.shape))
		unpool_delta, unpool_ref_delta = torch.chunk(unpool_, 2)

	unpool_delta_ = unpool_delta + unpool_ref_delta
	unpool_delta = torch.cat([unpool_delta_, unpool_delta_])
	idxs = torch.abs(delta_in) < 1e-7

	new_grad_inp = torch.where(idxs, grad_input[0], unpool_delta / delta_in)
	return (new_grad_inp,)


def deep_lift_shap(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: tuple | None = None,
	target: int = 0,
	batch_size: int = 32,
	references: Callable[..., Any] | torch.Tensor = dinucleotide_shuffle,
	n_shuffles: int = 20,
	return_references: bool = False,
	hypothetical: bool = False,
	warning_threshold: float = 0.001,
	additional_nonlinear_ops: dict | None = None,
	print_convergence_deltas: bool = False,
	raw_outputs: bool = False,
	only_warn: bool = False,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	random_state: int | numpy.random.RandomState | None = None,
	verbose: bool = False,
) -> torch.Tensor | AttributionReferencesResult:
	"""Calculate attributions for a set of sequences using DeepLIFT/SHAP.

	This function will calculate the DeepLIFT/SHAP attributions on a set of
	sequences given a model. These attributions have the additive property that
	the sum of the attributions is ~equal to the difference in prediction
	between the original sequence and the reference sequences.

	As an implementation note, the batch size refers to the number of
	example-reference pairs that are being run simultaneously. When the batch
	size is smaller than the number of references, multiple batches will be
	run per example and the attributions will only be averaged across the
	references after they have all been covered. You may want to do this if the
	model or examples are so large that only a few can fit in memory at a time.
	The result will be identical to if all examples could fit in memory and
	each batch contained all the references.

	Convergence deltas are calculated automatically for each example-reference
	pair. Theoretically, these should be zero, but may in practice just be a
	small number due to machine precision issues with non-linear models. If
	these deltas exceed a warning threshold, a non-terminating warning will be 
	issued to let you know that the deltas have been exceeded.

	NOTE: predictions MUST yield a `(batch_size, n_targets)` tensor, even if
	n_targets is 1. If your model yields something more complicated you must
	wrap the model in a small class that operates on the outputs in a manner
	that yields such a tensor, e.g., by slicing the output or summing along
	a relevant axis.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate attribution values
		for. 

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	target: int, optional
		The output of the model to calculate gradients/attributions for. This
		will index the last dimension of the predictions. Default is 0.

	batch_size: int, optional
		The number of sequence-reference pairs to pass through DeepLiftShap at
		a time. Importantly, this is not the number of elements in `X` that
		are processed simultaneously (alongside ALL their references) but the
		total number of `X`-`reference` pairs that are processed. This means
		that if you are in a memory-limited setting where you cannot process
		all references for even a single sequence simultaneously that the
		work is broken down into doing only a few references at a time. Default
		is 32.

	references: func or torch.Tensor, optional
		If a function is passed in, the function must accept `(X, n=...)` and
		(when `random_state` is not None) `(X, n=..., random_state=...)`. It is
		called with `n=1` per shuffle and once per `(example, shuffle_idx)` pair
		when seeded, or once per batch when not. It should return a tensor
		shaped `(batch, 1, *X.shape[1:])` (the second axis is squeezed off).
		The function should transform a sequence into some form of signal-null
		background, such as by shuffling it. If a torch.Tensor is passed in,
		that tensor must have shape `(len(X), n_shuffles, *X.shape[1:])`, in
		that for each sequence a number of shuffles are provided. Default is
		the function `dinucleotide_shuffle`.

	n_shuffles: int, optional
		The number of shuffles to use if a function is given for `references`.
		If a torch.Tensor is provided, this number is ignored. Default is 20.

	return_references: bool, optional
		Whether to return the references that were generated during this
		process. Only use if `references` is not a torch.Tensor. Default is
		False.

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position (`True`) or only for the character that is actually in the
		sequence (`False`). When `False`, the per-character attributions are
		multiplied by the one-hot encoded input so that only the observed
		character has a non-zero attribution at each position. Default is False.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the aggr_func being applied to them. Default 
		is 0.001. 

	additional_nonlinear_ops: dict or None, optional
		If additional nonlinear ops need to be added to the dictionary of
		operations that can be handled by DeepLIFT/SHAP, pass a dictionary here
		where the keys are class types and the values are the name of the
		function that handle that sort of class. Make sure that the signature
		matches those of `_nonlinear` and `_maxpool` above. This can also be
		used to overwrite the hard-coded operations by passing in a dictionary
		with overlapping key names. If None, do not add any additional 
		operations. Default is None.

	print_convergence_deltas: bool, optional
		Whether to print the convergence deltas for each example when using
		DeepLiftShap. Default is False.

	raw_outputs: bool, optional
		Whether to return the raw outputs from the method -- in this case,
		the multipliers for each example-reference pair -- or the processed
		attribution values. Default is False.

	only_warn: bool, optional
		Whether to only warn when a violation is recorded instead of raise a
		terminating error. This applies to input validation on `X` and (if
		passed) `references`; it does NOT suppress the runtime convergence-delta
		warnings emitted from the DeepLIFT computation itself. Default is False.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Default is None.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. Default
		is None.

	random_state: int or None, optional
		The random seed to use to ensure determinism. Must be an int (or None);
		the value is added to per-shuffle offsets when calling `references`, so
		`numpy.random.RandomState` instances are not supported here. If None,
		the process is not deterministic. Default is None.

	verbose: bool, optional
		Whether to display a progress bar. Default is False.


	Returns
	-------
	attributions: torch.tensor
		If `raw_outputs=False` (default), the attribution values with shape
		equal to `X`. If `raw_outputs=True`, the multipliers for each example-
		reference pair with shape equal to `(X.shape[0], n_shuffles, X.shape[1],
		X.shape[2])`.

	references: torch.tensor, optional
		The references used for each input sequence, with the shape
		(n_input_sequences, n_shuffles, 4, length). Only returned if
		`return_references = True`.
	"""

	_validate_input(X, "X", shape=(-1, -1, -1), ohe=True, only_warn=only_warn)

	if X.shape[0] == 0:
		raise ValueError("deep_lift_shap requires at least one example; got "
			"X with shape[0] == 0.")

	_NON_LINEAR_OPS = {
		torch.nn.ReLU: _nonlinear,
		torch.nn.ReLU6: _nonlinear,
		torch.nn.RReLU: _nonlinear,
		torch.nn.SELU: _nonlinear,
		torch.nn.CELU: _nonlinear,
		torch.nn.GELU: _nonlinear,
		torch.nn.SiLU: _nonlinear,
		torch.nn.Mish: _nonlinear,
		torch.nn.GLU: _nonlinear,
		torch.nn.ELU: _nonlinear,
		torch.nn.LeakyReLU: _nonlinear,
		torch.nn.Sigmoid: _nonlinear,
		torch.nn.Tanh: _nonlinear,
		torch.nn.Softplus: _nonlinear,
		torch.nn.Softshrink: _nonlinear,
		torch.nn.LogSigmoid: _nonlinear,
		torch.nn.PReLU: _nonlinear,
		torch.nn.MaxPool1d: _maxpool,
		torch.nn.MaxPool2d: _maxpool,
		torch.nn.Softmax: _softmax
	}

	device = _resolve_device(device)

	if dtype is None:
		try:
			dtype = next(model.parameters()).dtype
		except (StopIteration, AttributeError):
			dtype = torch.float32
	elif isinstance(dtype, str):
		dtype = getattr(torch, dtype)

	# Misc. set up for overriding operations

	if additional_nonlinear_ops is not None:
		for key, value in additional_nonlinear_ops.items():
			_NON_LINEAR_OPS[key] = value

	use_autocast = _autocast_supported(device, dtype)

	try:
		_orig_device = next(model.parameters()).device
	except StopIteration:
		_orig_device = None
	_was_training = model.training

	model.to(device).eval()

	try:
		for module in model.modules():
			module._NON_LINEAR_OPS = _NON_LINEAR_OPS

		try:
			model.apply(_register_hooks)
		except Exception as e:
			model.apply(_clear_hooks)
			raise(e)
	
		# Begin DeepLIFT procedure
	
		attributions, references_, Xi, rj, attr_ = [], [], [], [], []
		if isinstance(references, torch.Tensor):
			_validate_input(references, "references", shape=(X.shape[0], -1, X.shape[1], 
				X.shape[2]), ohe=True, allow_N=False, ohe_dim=-2, only_warn=only_warn)
			n_shuffles = references.shape[1]

		n, z = X.shape[0] * n_shuffles, 0

		for i in trange(n, disable=not verbose):
			Xi.append(i // n_shuffles)
			rj.append(i % n_shuffles)

			if len(Xi) == batch_size or i == (n-1):
				_X = X[Xi].cpu().type(dtype)
				_args = None if args is None else tuple([a[Xi].to(device).type(dtype)
					for a in args])

				# Handle reference sequences while ensuring that the same seed is
				# used for each shuffle even if not all shuffles are done in the
				# same batch.
				if isinstance(references, torch.Tensor):
					_references = references[Xi, rj]
				else:
					if random_state is None:
						_references = references(_X, n=1)[:, 0]
					else:
						_references = torch.cat([references(_X[j:j+1], n=1, 
							random_state=random_state+rj[j])[:, 0] 
								for j in range(len(_X))])

				_X = _X.to(device).type(dtype).requires_grad_()
				_references = _references.to(device).type(dtype).requires_grad_()

				# This next block is actually running DeepLIFT by concatenating the
				# batch of examples and the batch of references and running the
				# forward and backward passes that have been modified by the above
				# hooks. In a try-except block to make sure we remove hooks if an
				# error is raised. 
				try:
					X_ = torch.cat([_X, _references])

					if use_autocast:
						autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype)
					else:
						autocast_ctx = contextlib.nullcontext()

					# Calculate the gradients using the rescale rule
					with torch.autograd.set_grad_enabled(True):
						with autocast_ctx:
							if _args is not None:
								_args = (torch.cat([arg, arg]) for arg in _args)
								y = model(X_, *_args)[:, target]
							else:
								y = model(X_)[:, target]

							multipliers = torch.autograd.grad(y.sum(), _X)[0]

					# Check that the prediction-difference-from-reference is equal to
					# the sum of the attributions
					output_diff = torch.sub(*torch.chunk(y, 2))
					input_diff = torch.sum((_X - _references) * multipliers, 
						dim=(1, 2))
					convergence_deltas = abs(output_diff - input_diff)

					if torch.any(convergence_deltas > warning_threshold):
						warnings.warn("Convergence deltas too high: " +   
							str(convergence_deltas), RuntimeWarning)

					if print_convergence_deltas:
						print(convergence_deltas)

				except Exception as e:
					model.apply(_clear_hooks)
					raise(e)

				# If not returning the raw multipliers then apply the correction for
				# character encodings
				if raw_outputs == False:
					multipliers = hypothetical_attributions((multipliers,), (_X,), 
						(_references,))[0]

				# attr_ is a list where each element is a tensor for the multipliers
				# of one example-reference pair, so that once all references for an
				# example have been processed we can chunk them together.
				attr_.extend(list(multipliers.cpu().detach()))

				# When all references for a sequence have been calculated, remove
				# that block from the list of example-reference attributions and
				# add it to the final attribution list, averaging across references
				# if providing the processed results.
				while len(attr_) >= n_shuffles:
					attr_chunk = torch.stack(attr_[:n_shuffles])

					if raw_outputs == False:
						attr_chunk = attr_chunk.mean(dim=0)
						if not hypothetical:
							attr_chunk *= X[z].cpu()

					attributions.append(attr_chunk)
					attr_ = attr_[n_shuffles:]
					z += 1

				if return_references:
					references_.extend(list(_references.cpu().detach()))

				Xi, rj = [], []



		attributions = torch.stack(attributions)

		if return_references:
			references_ = torch.cat(references_).reshape(X.shape[0], n_shuffles,
				*X.shape[1:])
			return AttributionReferencesResult(
				attributions=attributions, references=references_)
		return attributions
	finally:
		model.apply(_clear_hooks)
		for module in model.modules():
			if hasattr(module, "_NON_LINEAR_OPS"):
				del module._NON_LINEAR_OPS

		if _was_training:
			model.train()
		if _orig_device is not None and _orig_device != device:
			model.to(_orig_device)


def _captum_deep_lift_shap(
	model: torch.nn.Module,
	X: torch.Tensor,
	args: tuple | None = None,
	target: int = 0,
	batch_size: int = 32,
	references: Callable[..., Any] | torch.Tensor = dinucleotide_shuffle,
	n_shuffles: int = 20,
	return_references: bool = False,
	hypothetical: bool = False,
	device: str | torch.device | None = None,
	random_state: int | numpy.random.RandomState | None = None,
	verbose: bool = False,
) -> torch.Tensor | AttributionReferencesResult:
	"""Calculate attributions using captum's DeepLiftShap and a given model.

	This function will calculate DeepLift/Shap attributions on a set of
	sequences by delegating to captum's `DeepLiftShap` and using a user-
	supplied (or default `dinucleotide_shuffle`) reference function or
	pre-computed reference tensor, the same `references` contract used by
	`deep_lift_shap`. It does NOT make any assumption about the structure of
	the model output (no BPNet logits/counts split) and it does NOT generate
	GC-matched negatives.

	This is an internal/debugging function that is mostly meant to be used to
	check for differences with the `deep_lift_shap` method.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(-1, len(alphabet), length)
		A set of one-hot encoded sequences to calculate attribution values
		for. 

	args: tuple or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	target: int, optional
		The output of the model to calculate gradients/attributions for. This
		will index the last dimension of the predictions. Default is 0.

	batch_size: int, optional
		The number of sequence-reference pairs to pass through DeepLiftShap at
		a time. Importantly, this is not the number of elements in `X` that
		are processed simultaneously (alongside ALL their references) but the
		total number of `X`-`reference` pairs that are processed. This means
		that if you are in a memory-limited setting where you cannot process
		all references for even a single sequence simultaneously that the
		work is broken down into doing only a few references at a time. Default
		is 32.

	references: func or torch.Tensor, optional
		If a function is passed in, the function must accept `(X, n=...,
		random_state=...)`. It is called once per example with `n=n_shuffles`
		and should return a tensor shaped `(1, n_shuffles, *X.shape[1:])`
		(the leading axis is indexed off). The function should transform a
		sequence into some form of signal-null background, such as by
		shuffling it. If a torch.Tensor is passed in, that tensor must have
		shape `(len(X), n_shuffles, *X.shape[1:])`, in that for each sequence
		a number of shuffles are provided. Default is the function
		`dinucleotide_shuffle`.

	n_shuffles: int, optional
		The number of shuffles to use if a function is given for `references`.
		If a torch.Tensor is provided, this number is ignored. Default is 20.

	return_references: bool, optional
		Whether to return the references that were generated during this
		process. Only use if `references` is not a torch.Tensor. Default is
		False.

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position (`True`) or only for the character that is actually in the
		sequence (`False`). When `False`, the per-character attributions are
		multiplied by the one-hot encoded input so that only the observed
		character has a non-zero attribution at each position. Default is False.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. Default
		is None.

	random_state: int or None, optional
		The random seed to use to ensure determinism. Passed through to the
		`references` callable; `numpy.random.RandomState` instances are not
		supported. If None, the process is not deterministic. Default is None.

	verbose: bool, optional
		Whether to display a progress bar. Default is False.


	Returns
	-------
	attributions: torch.tensor
		The attributions calculated for each input sequence, with the same
		shape as the input sequences.

	references: torch.tensor, optional
		The references used for each input sequence, with the shape
		(n_input_sequences, n_shuffles, 4, length). Only returned if
		`return_references = True`.
	"""

	from captum.attr import DeepLiftShap as CaptumDeepLiftShap

	device = _resolve_device(device)
	model = model.to(device).eval()

	attributions = []
	references_ = []
	with torch.no_grad():
		for i in trange(len(X), disable=not verbose):
			_X = X[i:i+1].to(device)
			_args = None if args is None else tuple([a[i:i+1].to(device) 
				for a in args])

			# Calculate references
			if isinstance(references, torch.Tensor):
				_references = references[i].to(device)
			else:
				_references = references(_X, n=n_shuffles, 
					random_state=random_state)[0].to(device)
						
			attr = CaptumDeepLiftShap(model).attribute(_X, _references, 
				target=target, additional_forward_args=_args, 
				custom_attribution_func=hypothetical_attributions)

			if not hypothetical:
				attr = (attr * _X)
			
			if return_references:
				references_.append(_reference.unsqueeze(0))

			attributions.append(attr.cpu())

	attributions = torch.cat(attributions)

	if return_references:
		return AttributionReferencesResult(
			attributions=attributions, references=torch.cat(references_))
	return attributions