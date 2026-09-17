# screen.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any

import numpy
import torch

from ..utils import _cast_as_tensor
from ..utils import random_one_hot

from ..predict import predict


def screen(
	model: torch.nn.Module,
	shape: tuple[int, ...],
	y: torch.Tensor | list[torch.Tensor] | None = None,
	loss: Callable[..., Any] = torch.nn.MSELoss(reduction='none'),
	tol: float = 1e-3,
	max_iter: int = -1,
	args: tuple | None = None,
	n_best: int = 1,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	batch_size: int = 32,
	func: Callable[..., Any] = random_one_hot,
	additional_func_kwargs: dict | None = None,
	dtype: str | torch.dtype | None = None,
	device: str | torch.device | None = None,
	random_state: int | numpy.random.RandomState | None = None,
	verbose: bool = False,
) -> torch.Tensor:
	"""Screen randomly generated sequences and choose the best one.

	Potentially, the conceptually simplest method for design is to randomly
	generate a batch of examples and evaluate them using the provided model,
	keeping only the `n_best` top hits according to the loss function. This is
	called "screening", as one is "screening" a large pool of random potential
	designs for activity and keeping only those that appear good according to some
	loss function. 

	Although this function will likely be slow since each batch is independent
	from the others, i.e., you are not guaranteed to be getting closer to a
	goal with each step, you may be surprised by how good the generations are.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model to use for making predictions. These models can take in
		any number of inputs and make any number of outputs. The additional
		inputs must be specified in the `args` parameter.

	shape: tuple
		Dimensions for the randomly generated sequences, excluding the batch
		dimension. For a model expecting an input like (32, 4, 2114), where
		32 is the batch size, `shape` should be (4, 2114). 

	y: torch.Tensor or list of torch.Tensors or None
		A tensor or list of Tensors providing the desired output from the model.
		The type and shape must be compatible with the provided loss function
		and comparable to the output from `model`. Each tensor should have a
		shape of (1, n) where n is the number of outputs from the model. The 
		first dimension is 1 to make broadcasting work correctly. If None,
		simply choose the edit that yields the strongest response from the
		model. Default is None.

	loss: function, optional
		This function must take in `y` and `y_hat` where `y` is the desired
		output from the model and `y_hat` is the current prediction from the
		model given the substitutions. By default, this is the 
		torch.nn.MSELoss().

	tol: float, optional
		A threshold on the loss below which the screening procedure terminates.
		Termination requires the loss of the *worst* kept candidate (i.e. the
		`n_best`-th best so far) to fall below `tol` — when `n_best > 1` the
		current best may have been below `tol` for many iterations before this
		condition triggers. Default is 1e-3.

	max_iter: int, optional
		The maximum number of iterations to run before terminating the procedure.
		Set to -1 for no limit. Default is -1.

	args: tuple or list or None, optional
		An optional set of additional arguments to pass into the model. If
		provided, each element in the tuple or list is one input to the model
		and the element must be formatted to be the same batch size as `X`. If
		None, no additional arguments are passed into the forward function.
		Default is None.

	n_best: int, optional
		The number of sequences to return at the end, ranked from the lowest loss
		to the highest loss. Setting to 1 means only return the very best sequence
		observed across all generation batches. Default is 1.

	batch_size: int, optional
		The number of sequences to generate (via `func`) and evaluate per
		iteration. This controls the size of each generation/screening batch;
		it is NOT forwarded to `predict`'s own batch_size (which retains its
		default). Default is 32.

	func: function, optional
		The function to use to generate sequences. The signature of this function
		must be that it takes in a tuple of the shape of the batch to generate, e.g. 
		(32, 4, 2114), and also a random state. Default is `random_one_hot`. 

	additional_func_kwargs: dict or None, optional
		Additional named arguments to pass into the function when it is called.
		This is provided as an alternate path to route arguments into the
		function in case they overlap, name-wise, with those in this function,
		or if you want to be absolutely sure that the arguments are making
		their way into the function. The dict is not modified in place. Default
		is None.

	dtype: str or torch.dtype or None, optional
		The dtype to use with mixed precision autocasting. If None, use the dtype of
		the *model*. This allows you to use int8 to represent large data sets and
		only convert batches to the higher precision, saving memory. Default is None.

	device: str or torch.device or None, optional
		The device to move the model and batches to when making predictions. If
		None, use CUDA when available and fall back to CPU otherwise. Default
		is None.

	random_state: int or None, optional
		The random seed to use to ensure determinism of the generation function.
		The seed is incremented by 1 at the end of each iteration so successive
		iterations draw different sequences while remaining reproducible. If
		None, not deterministic. Default is None.

	verbose: bool, optional
		Whether to display a progress bar during predictions. Default is False.


	Returns
	-------
	X: torch.Tensor, shape=(n_best, len(alphabet), length)
		The screened examples with the lowest loss.
	"""

	if y is None:
		y = [9_999_999]

	y = _cast_as_tensor(y)
	additional_func_kwargs = dict(additional_func_kwargs or {})

	pq = []
	iteration = 0

	while True:
		X = func((batch_size, *shape), random_state=random_state,
			**additional_func_kwargs)

		y_hat = predict(model, X, dtype=dtype, device=device, args=args)

		current_loss = -loss(y, y_hat).numpy(force=True).sum(axis=1)
		for i in range(batch_size):
			l = current_loss[i]
			entry = [l, X[i]]

			if len(pq) < n_best:
				heapq.heappush(pq, entry)
			elif l > pq[0][0]:
				heapq.heappushpop(pq, entry)
			else:
				continue

			if verbose:
				print("Adding element with loss {:4.4}".format(l))

		iteration += 1
		if -pq[0][0] < tol or iteration == max_iter:
			break
		
		if random_state is not None:
			random_state += 1
    
	Xs = [heapq.heappop(pq)[1] for i in range(n_best)]
	X = torch.flip(torch.stack(Xs), dims=(0,))
	return X
