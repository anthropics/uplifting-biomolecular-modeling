# greedy_substitution.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from tqdm import tqdm

from ..utils import _cast_as_tensor
from ..utils import one_hot_encode
from ..utils import reverse_complement as rc

from ..ersatz import substitute
from ..predict import predict

from ._substitute import _fast_tile_substitute


def greedy_substitution(
	model: torch.nn.Module,
	X: torch.Tensor,
	y: torch.Tensor | list[torch.Tensor] | None = None,
	motifs: list[str] | None = None,
	loss: Callable[..., Any] = torch.nn.MSELoss(reduction='none'),
	reverse_complement: bool = True,
	input_mask: torch.Tensor | None = None,
	output_mask: torch.Tensor | None = None,
	tol: float = 1e-3,
	max_iter: int = -1,
	args: tuple | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	batch_size: int = 32,
	device: str | torch.device | None = None,
	verbose: bool = False,
) -> torch.Tensor:
	"""Greedily add motifs to achieve a desired goal. 

	This design function will greedily add motifs to achieve a desired output
	from the model. Each round, the function will iterate through all possible
	motifs, substitute each one with the given spacing, and keep the one whose
	loss function is the smallest. This process will continue until either the
	maximum number of iterations is reached (at which point, `max_iter` motifs
	will have been inserted into the sequence) or the loss falls below `tol`.

	Accordingly, the choice of loss function and desired output from the model
	is crucial for good design. Usually, the loss function can be Euclidean
	distance, but for models with more complex outputs or for subtle design
	tasks one may want to use something else, such as Jensen-Shannon divergence.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	X: torch.tensor, shape=(1, len(alphabet), length)
		A one-hot encoded sequence to use as the base for design. This must be
		a single sequence and has the first dimension for broadcasting reasons.
		A first dimension other than 1 raises a `ValueError`; loop over the
		examples, passing `X[i:i+1]` to each call, to design more than one.

	y: torch.Tensor or list of torch.Tensors or None
		A tensor or list of Tensors providing the desired output from the model.
		The type and shape must be compatible with the provided loss function
		and comparable to the output from `model`. Each tensor should have a
		shape of (1, n) where n is the number of outputs from the model. The 
		first dimension is 1 to make broadcasting work correctly. If None,
		simply choose the edit that yields the strongest response from the
		model. Default is None.

	motifs: list of strings or None
		A list of strings where each string is a motif that can be inserted into
		the sequence. These strings will be one-hot encoded according to the
		provided alphabet. If None, use the provided alphabet as the motifs to
		only change one character at a time. Default is None.

	loss: function, optional
		This function must take in `y` and `y_hat` where `y` is the desired
		output from the model and `y_hat` is the current prediction from the
		model given the substitutions. By default, this is the 
		torch.nn.MSELoss().

	reverse_complement: bool, optional
		Whether to augment the provided list of motifs with their reverse
		complements. This will double the runtime. Default is True.

	input_mask: torch.Tensor or None, optional
		A mask on input positions that can be the start of substitution. Any
		motif can be substituted in starting at each allowed position even if
		the contiguous span of the mask is shorter than the motif. True means
		that a motif can be substituted in starting at that position and False
		means that it cannot be. Default is None.

	output_mask: torch.Tensor or None, optional
		A mask on the outputs from the model to consider. True means to include
		the outputs in the loss, False means to exclude those outputs from the
		loss. If None, use all outputs. Default is None.

	tol: float, optional
		A threshold on the amount of improvement necessary according to loss,
		where the procedure will stop once the improvement is below. Default
		is 1e-3.

	max_iter: int, optional
		The maximum number of iterations to run before terminating the
		procedure. The loop condition is `iteration < max_iter`, so any
		non-positive value (including the `-1` default) causes the loop body
		to never execute and the original `X` to be returned unchanged. Pass
		a positive integer (or a large sentinel such as `10**9`) for actual
		iteration. Default is -1.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. This is not necessary or used if a one-hot encoded tensor is
		provided for the motif. Default is ['A', 'C', 'G', 'T'].

	batch_size: int, optional
		The number of examples to make predictions for at a time. Default is 32.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. Default
		is None.

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	X: torch.Tensor, shape=(-1, len(alphabet), length)
		The edited sequence.
	"""

	tic = time.time()
	iteration = 0

	if X.shape[0] != 1:
		raise ValueError("greedy_substitution designs a single sequence at a "
			"time and requires X.shape[0] == 1; got X with shape[0] == "
			f"{X.shape[0]}. Loop over the examples, passing X[i:i+1] to each "
			"call.")

	X = torch.clone(X)
	y_orig = predict(model, X, args=args, batch_size=batch_size, device=device,
		verbose=False)

	if motifs is None:
		motifs = alphabet
		reverse_complement = False
	
	if reverse_complement:
		motifs = motifs + [rc(motif) for motif in motifs]

	if output_mask is None:
		output_mask = torch.ones(y_orig.shape[1], dtype=bool)

	if input_mask is None:
		input_mask = torch.ones(X.shape[-1], dtype=bool)

	if y is not None:
		y = _cast_as_tensor(y)
		if y.ndim == 1:
			y = y.unsqueeze(0)

		loss_prev = loss(y[:, output_mask], y_orig[:, output_mask]).mean()
	else:
		loss_prev = -y_orig[:, output_mask].mean()
	
	
	loss_orig = loss_prev

	if verbose:
		print(("Iteration 0 -- Loss: {:4.4}, Improvement: N/A, Idx: N/A, " +
			"Time (s): 0s").format(loss_prev, time.time() - tic))

	while iteration < max_iter:
		tic = time.time()
		best_improvement, best_motif_idx, best_pos = 0, -1, -1
		
		for idx, motif in enumerate(tqdm(motifs, disable=not verbose)):
			motif_ohe = one_hot_encode(motif, alphabet=alphabet).numpy()
			
			input_mask_ = torch.clone(input_mask)
			if len(motif) > 1:
				input_mask_[-len(motif)+1:] = False
			input_idxs = torch.where(input_mask_ == True)[0].numpy()

			X_ = X.float().repeat(input_idxs.shape[0], 1, 1).numpy(force=True)
			_fast_tile_substitute(X_, motif_ohe, input_idxs)
			X_ = torch.from_numpy(X_).to(X.dtype)

			y_hat = predict(model, X_, args=args, batch_size=batch_size, 
				device=device, verbose=False)

			if y is not None:
				loss_curr = loss(
					y[:, output_mask].expand_as(y_hat[:, output_mask]), 
					y_hat[:, output_mask],
				).mean(dim=tuple(range(1, len(y_hat.shape))))
			else:
				loss_curr = -y_hat[:, output_mask].mean(
					dim=tuple(range(1, len(y_hat.shape))))
			
			pos = loss_curr.argmin()
			loss_curr = loss_curr[pos]
			
			improvement = loss_prev - loss_curr
			if improvement > best_improvement:
				best_improvement = improvement
				best_motif_idx = idx
				best_pos = input_idxs[pos]
				best_loss = loss_curr


		if best_motif_idx != -1:
			X = substitute(X, motifs[best_motif_idx], start=best_pos, 
				alphabet=alphabet)
			loss_prev = best_loss

			if verbose:
				print(("Iteration {} -- Loss: {:4.4}, Improvement: {:4.4}, " + 
					"Motif Idx: {}, Pos Idx: {}, Time (s): {:4.4}").format(
						iteration+1, best_loss, best_improvement, 
						best_motif_idx, best_pos, time.time() - tic))

		if best_improvement <= tol:
			break

		iteration += 1

	return X
