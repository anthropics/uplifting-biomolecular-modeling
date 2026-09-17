# attribute.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


from bpnetlite.bpnet import _ProfileLogitScaling
from bpnetlite.chrombpnet import _Log, _Exp

from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.deep_lift_shap import deep_lift_shap as t_deep_lift_shap
from tangermeme.deep_lift_shap import _nonlinear


def deep_lift_shap(model, X, args=None, target=0,  batch_size=32,
	references=dinucleotide_shuffle, n_shuffles=20, return_references=False, 
	hypothetical=False, warning_threshold=0.001, additional_nonlinear_ops=None,
	print_convergence_deltas=False, raw_outputs=False, device='cuda', 
	random_state=None, verbose=False):
	"""A wrapper that registers Chrom/BPNet's custom non-linearities.

	This function is just a wrapper for tangermeme's deep_lift_shap function
	except that it automatically registers the layers that are necessary for
	using BPNet models. Specifically, it registers a scaling that is necessary
	for calculating the profile attributions and also registers the logsumexp
	operation for counts when using the full ChromBPNet model.

	Other than automatically registering the non-linearities, this wrapper does
	not modify the tangermeme outputs or alter the inputs in any way. It is
	simply for convenience so you do not need to reach into bpnet-lite's
	internals each time you want to calculate attributions.


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
		If a function is passed in, this function is applied to each sequence
		with the provided random state and number of shuffles. This function
		should serve to transform a sequence into some form of signal-null
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
		position or only for the character that is actually at the sequence.
		Practically, whether to return the returned attributions from captum
		with the one-hot encoded sequence. Default is False.

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

	device: str or torch.device, optional
		The device to move the model and batches to when making predictions. If
		set to 'cuda' without a GPU, this function will crash and must be set
		to 'cpu'. Default is 'cuda'. 

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 

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

	return t_deep_lift_shap(model=model, X=X, args=args, target=target, 
		batch_size=batch_size, references=references, n_shuffles=n_shuffles, 
		return_references=return_references, hypothetical=hypothetical, 
		warning_threshold=warning_threshold,
		additional_nonlinear_ops={
			_ProfileLogitScaling: _nonlinear,
			_Log: _nonlinear,
			_Exp: _nonlinear
		},
		print_convergence_deltas=print_convergence_deltas, 
		raw_outputs=raw_outputs, device=device, random_state=random_state, 
		verbose=verbose)
