# beam_substitution.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import time
import heapq
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


def beam_substitution(
	model: torch.nn.Module,
	X: torch.Tensor,
	y: torch.Tensor | list[torch.Tensor] | None = None,
	motifs: list[str] | None = None,
	loss: Callable[..., Any] = torch.nn.MSELoss(reduction='none'),
	reverse_complement: bool = True,
	input_mask: torch.Tensor | None = None,
	output_mask: torch.Tensor | None = None,
	beam_size: int = 4,
	n_best: int = 1,
	tol: float = 1e-3,
	max_iter: int = -1,
	args: tuple | None = None,
	alphabet: list[str] = ['A', 'C', 'G', 'T'],
	batch_size: int = 32,
	device: str | torch.device | None = None,
	verbose: bool = False,
) -> torch.Tensor:
	"""Beam search over motif substitutions to achieve a desired goal.

	This is a generalization of `greedy_substitution`. Rather than committing
	to the single best edit each round, beam search keeps the `beam_size` best
	complete sequences (the "beam") and expands all of them. The classic
	difficulty with applying beam search to a sequence-to-function model is that
	the model only produces a meaningful output once a full, fixed-length input
	is filled, so there is no natural notion of scoring a partially-built
	sequence the way one scores a partial sentence. This implementation sidesteps
	that entirely: every candidate in the beam is always a complete, fixed-length
	sequence, because each step *substitutes* a motif into an existing sequence
	rather than growing one. What is searched over is therefore not positions in
	a growing string but trajectories through edit-space.

	Each round, every beam member is expanded by tiling every motif at every
	allowed position (exactly as `greedy_substitution` does for its single
	sequence), all resulting complete sequences are scored by their absolute
	loss, and the global top-`beam_size` are kept as the next beam. The current
	beam members are themselves carried forward as candidates, so the beam never
	regresses. Identical sequences are de-duplicated before pruning so that the
	beam does not collapse onto a single sequence. Setting `beam_size=1` recovers
	`greedy_substitution`; larger beams hedge across multiple trajectories and
	can recover good multi-edit combinations that the greedy method prunes away
	after a locally-suboptimal first edit.

	As with `greedy_substitution`, the choice of loss function and desired output
	is crucial. Usually the loss can be Euclidean distance, but for models with
	more complex outputs one may want something else, such as Jensen-Shannon
	divergence.


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
		simply choose the edits that yield the strongest response from the
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

	beam_size: int, optional
		The number of complete sequences to keep in the beam each round. Setting
		this to 1 recovers `greedy_substitution`. Larger values explore more
		trajectories at a cost that scales linearly with `beam_size`. Default
		is 4.

	n_best: int, optional
		The number of sequences to return at the end, ranked from the lowest
		loss to the highest loss. Must be no larger than `beam_size`; if larger,
		it is clamped to the number of distinct sequences in the final beam.
		Default is 1.

	tol: float, optional
		A threshold on the amount of improvement necessary according to loss,
		where the procedure will stop once the round-over-round improvement of
		the best beam member is below it. Default is 1e-3.

	max_iter: int, optional
		The maximum number of iterations (edits) to run before terminating the
		procedure. Set to -1 for no limit. Default is -1.

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
	X: torch.Tensor, shape=(n_best, len(alphabet), length)
		The designed sequences, ranked from lowest loss to highest loss.
	"""

	tic = time.time()
	iteration = 0

	if X.shape[0] != 1:
		raise ValueError("beam_substitution designs a single sequence at a "
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

	def _seq_loss(y_hat):
		if y is not None:
			return loss(
				y[:, output_mask].expand_as(y_hat[:, output_mask]),
				y_hat[:, output_mask],
			).mean(dim=tuple(range(1, len(y_hat.shape))))
		return -y_hat[:, output_mask].mean(dim=tuple(range(1, len(y_hat.shape))))

	# Each beam member is a (loss, sequence) pair where the sequence is a
	# complete, fixed-length, one-hot encoded tensor of shape (1, alphabet, len).
	beam = [(float(_seq_loss(y_orig)[0]), X)]
	loss_prev = beam[0][0]

	if verbose:
		print(("Iteration 0 -- Loss: {:4.4}, Improvement: N/A, Idx: N/A, " +
			"Time (s): 0s").format(loss_prev, time.time() - tic))

	while True:
		if iteration == max_iter:
			break

		tic = time.time()

		# The heap keeps the `beam_size` candidates with the highest score
		# (= lowest loss) seen this round. Entries are
		# (score, counter, beam_idx, motif_idx, pos); the counter is a unique
		# tiebreaker so the heap never compares the trailing fields, keeping
		# selection deterministic. motif_idx == -1 means "keep this beam member
		# unchanged", which lets the beam carry good sequences forward.
		pq = []
		counter = 0

		def _consider(score, beam_idx, motif_idx, pos):
			nonlocal counter
			entry = (score, counter, beam_idx, motif_idx, pos)
			counter += 1
			if len(pq) < beam_size:
				heapq.heappush(pq, entry)
			elif score > pq[0][0]:
				heapq.heappushpop(pq, entry)

		for beam_idx, (b_loss, Xb) in enumerate(beam):
			_consider(-b_loss, beam_idx, -1, -1)

			for idx, motif in enumerate(tqdm(motifs, disable=not verbose)):
				motif_ohe = one_hot_encode(motif, alphabet=alphabet).numpy()

				input_mask_ = torch.clone(input_mask)
				if len(motif) > 1:
					input_mask_[-len(motif)+1:] = False
				input_idxs = torch.where(input_mask_ == True)[0].numpy()

				X_ = Xb.float().repeat(input_idxs.shape[0], 1, 1).numpy(force=True)
				_fast_tile_substitute(X_, motif_ohe, input_idxs)
				X_ = torch.from_numpy(X_).to(X.dtype)

				y_hat = predict(model, X_, args=args, batch_size=batch_size,
					device=device, verbose=False)

				loss_curr = _seq_loss(y_hat)

				# Only the `beam_size` lowest-loss positions for this motif can
				# ever survive a beam of this size, so prune before pushing.
				k = min(beam_size, loss_curr.shape[0])
				top = torch.topk(-loss_curr, k).indices
				for pos in top:
					_consider(float(-loss_curr[pos]), beam_idx, idx,
						int(input_idxs[pos]))

		# Reconstruct sequences best-first and de-duplicate identical ones so
		# the beam keeps `beam_size` distinct trajectories.
		entries = sorted(pq, key=lambda e: (-e[0], e[1]))
		new_beam = []
		seen = set()
		for score, _, beam_idx, motif_idx, pos in entries:
			if motif_idx == -1:
				seq = torch.clone(beam[beam_idx][1])
			else:
				seq = substitute(beam[beam_idx][1], motifs[motif_idx],
					start=pos, alphabet=alphabet)

			key = seq.numpy(force=True).tobytes()
			if key in seen:
				continue

			seen.add(key)
			new_beam.append((-score, seq))

		beam = new_beam
		loss_curr = beam[0][0]
		improvement = loss_prev - loss_curr

		if verbose:
			print(("Iteration {} -- Loss: {:4.4}, Improvement: {:4.4}, " +
				"Beam: {}, Time (s): {:4.4}").format(iteration+1, loss_curr,
					improvement, len(beam), time.time() - tic))

		loss_prev = loss_curr
		iteration += 1

		if improvement <= tol:
			break

	n_best = min(n_best, len(beam))
	return torch.cat([seq for _, seq in beam[:n_best]], dim=0)
