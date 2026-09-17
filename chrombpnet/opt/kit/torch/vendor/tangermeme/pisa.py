# pisa.py
# Based on the algorithm proposed by Charles McAnany et al.
# https://www.biorxiv.org/content/10.1101/2025.04.07.647613v2
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy
import torch

from tqdm import trange

from ._compat import _resolve_device
from .ersatz import dinucleotide_shuffle
from .results import AttributionReferencesResult

from tangermeme.predict import predict
from tangermeme.utils import _validate_input

from tangermeme.deep_lift_shap import _nonlinear, _maxpool, _softmax
from tangermeme.deep_lift_shap import _clear_hooks, _register_hooks
from tangermeme.deep_lift_shap import hypothetical_attributions


def pisa(
    model: torch.nn.Module,
    X: torch.Tensor,
    args: tuple | None = None,
    batch_size: int = 32,
    references: Callable[..., Any] | torch.Tensor = dinucleotide_shuffle,
    n_shuffles: int = 20,
    return_references: bool = False,
    hypothetical: bool = False,
    warning_threshold: float = 0.001,
    additional_nonlinear_ops: dict | None = None,
    print_convergence_deltas: bool = False,
    raw_outputs: bool = False,
    device: str | torch.device | None = None,
    random_state: int | numpy.random.RandomState | None = None,
    verbose: bool = False,
) -> torch.Tensor | AttributionReferencesResult:
    """An implementation of Pairwise Influence by Sequence Attribution (PISA).
    
    PISA is a method for deciphering how each input in an example influences
    each output. For sequence-based machine learning models this usually means
    how each nucleotide influences the predictions made at each output. This
    can be useful for understanding precisely what parts of a predicted profile
    are influenced by the presence of certain motifs or other changes in
    nucleotide composition.
    
    In practice, PISA involves running DeepLIFT/SHAP once for each output
    position. Given a fixed input and model, one might suspect that the
    Jacobian could be calculated simply to speed up DeepLIFT/SHAP, but this
    seems to not be the case. This is, in part, due to DeepLIFT/SHAP overriding
    some of the gradient operations in ways that do not play nicely with some
    of PyTorch's built-in functions.
    
    
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

    device: str or torch.device or None, optional
        The device to move the model and batches to when making predictions. If
        None, use CUDA when available and fall back to CPU otherwise. Default
        is None.

    random_state: int or None or numpy.random.RandomState, optional
        The random seed to use to ensure determinism. If None, the
        process is not deterministic. Default is None. 

    verbose: bool, optional
        Whether to display a progress bar. Default is False.


    Returns
    -------
    attributions: torch.tensor
        If `raw_outputs=False` (default), the per-output attribution values
        with shape `(X.shape[0], n_outputs, X.shape[1], X.shape[2])`. If
        `raw_outputs=True`, the multipliers for each example-reference pair
        with shape `(X.shape[0], n_shuffles, n_outputs, X.shape[1], X.shape[2])`.

    references: torch.tensor, optional
        The references used for each input sequence, with the shape
        (n_input_sequences, n_shuffles, 4, length). Only returned if
        `return_references = True`.

    Notes
    -----
    `n_outputs` is inferred from the *last* axis of the model's output
    (`predict(model, X[:1]).shape[-1]`). Profile-style models that return
    `(batch, n_tasks, length)` will therefore be attributed over `length`
    only — to attribute over tasks instead, wrap the model in a small
    adapter that transposes or flattens the output.

    The returned `attributions` and `references` tensors are produced via
    `torch.stack` over per-example accumulators and may live on the input
    device rather than CPU in some return paths; explicit `.cpu()` is
    advisable when consuming results on a non-CUDA host.
    """

    _validate_input(X, "X", shape=(-1, -1, -1), ohe=True)

    if X.shape[0] == 0:
        raise ValueError("pisa requires at least one example; got X with "
            "shape[0] == 0.")

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

    # Misc. set up for overriding operations

    if additional_nonlinear_ops is not None:
        for key, value in additional_nonlinear_ops.items():
            _NON_LINEAR_OPS[key] = value

    device = _resolve_device(device)
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

        # Begin PISA procedure    
        attributions, references_ = [], [] 
        if isinstance(references, torch.Tensor):
            _validate_input(references, "references", shape=(X.shape[0], -1, X.shape[1], 
                X.shape[2]), ohe=True, allow_N=False, ohe_dim=-2)
            n_shuffles = references.shape[1]

        _probe_args = None if args is None else tuple(a[:1] for a in args)
        n_outputs = predict(model, X[:1], args=_probe_args, device=device).shape[-1]

        # Loop over each of the examples
        for i in trange(len(X), disable=not verbose):
            _X = X[i:i+1].to(device).requires_grad_()

            # Either use reference sequences if provided or generate a set of them
            # for this example being analyzed.
            if isinstance(references, torch.Tensor):
                _references = references[i]
            else:
                _references = references(_X.cpu(), n=n_shuffles, 
                    random_state=random_state)[0]

            _references = _references.to(device).requires_grad_()
            if return_references:
                references_.append(_references)

            # Pull out the additional arguments for this example, if additional
            # arguments are being provided
            _args = None if args is None else tuple([a[i].to(device) 
                for a in args])

            multipliers = []

            # Loop over all of the shuffles for each example
            for j in range(n_shuffles):
                # For each output explained, the input and the reference will be
                # the same, so extract those single elements and expand them to
                # be the batch size.

                x = _X.expand(batch_size, -1, -1)
                r = _references[j:j+1].expand(batch_size, -1, -1)
                xr = (x[:1] - r[:1])
            
                X_ = torch.cat([x, r])
            
                with torch.autograd.set_grad_enabled(True):
                    if _args is not None:
                        # Materialize into a tuple and bind to a fresh name so
                        # `_args` is not overwritten by an exhausted generator
                        # for the next shuffle iteration.
                        _args_batched = tuple(torch.cat([arg, arg]) for arg in _args)
                        y = model(X_, *_args_batched)
                    else:
                        y = model(X_)

                    _multipliers = []
                    edge_size = n_outputs % batch_size

                    # Loop over all of the output positions
                    for k in range(0, n_outputs, batch_size):
                        b = min(batch_size, n_outputs - k)
 
                        rows = torch.cat([torch.arange(b), torch.arange(b)+batch_size])
                        cols = torch.arange(k, k+b).repeat(2)
                        y_hat = y[rows, cols].sum()
                    
                        multipliers_ = torch.autograd.grad(y_hat, x, retain_graph=True)[0]
                    
                        # Check that the prediction-difference-from-reference is equal to
                        # the sum of the attributions                    
                        output_diff = torch.sub(*torch.chunk(y[rows, cols], 2))
                        input_diff = torch.sum(xr * multipliers_[:b], dim=(1, 2))
                                       
                        convergence_deltas = abs(output_diff - input_diff)
                        if torch.any(convergence_deltas > warning_threshold):
                            warnings.warn("Convergence deltas too high: " +   
                                str(convergence_deltas), RuntimeWarning)

                        if print_convergence_deltas:
                            print(convergence_deltas)
                    
                        if not raw_outputs:
                            multipliers_ = hypothetical_attributions(
                                (multipliers_[:b],), 
                                (x[:b],), 
                                (r[:b],)
                            )[0].cpu().detach()    
                                       
                        _multipliers.append(multipliers_[:b])                  

                # Discard the accumulated graph
                torch.autograd.grad(y[:1, :1].sum(), x, retain_graph=False)[0]

                # Keep the multipliers across each output
                multipliers.append(torch.cat(_multipliers, dim=0))

            attr = torch.stack(multipliers, dim=0)
            if not raw_outputs:
                attr = attr.mean(dim=0)
                if not hypothetical:
                    attr = attr * _X.cpu()
        
            attributions.append(attr)


        attributions = torch.stack(attributions).detach()

        if return_references:
            return AttributionReferencesResult(attributions=attributions,
                references=torch.stack(references_).detach())
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
