"""THE FORWARD CALL (``kitlib/forward.py``): one switch, ``KIT_FWD`` (``1`` = the kit's forward call; default ``0`` = the stock call,
untouched).

The stock call is ``SeqNN.__call__`` (baskerville seqnn.py:1046-1060): ``preds = model(x).numpy().astype(dtype)`` — the Keras ensemble
model executed EAGERLY on the (ref, alt) pair, op by op from Python, the device-to-host copy, then a SECOND float32 -> float32 host copy
of the (2, L, T) prediction array (``astype`` copies by default).

``KIT_FWD=1`` (stamped ``kit_fwd = graph_copy_free``) changes how that same computation is dispatched and returned, never what it computes:

* graph mode from the second call on. The FIRST call of the process is the stock's eager call, verbatim — it is also the call in which
  cuDNN selects its convolution algorithms, so the selection happens exactly as in a stock process and is cached per process. The wrapper
  then traces the same Keras model ONCE with ``tf.function(model, jit_compile=False)`` — no XLA, no fusion, no optimizer option changed:
  the same ops with the same attributes on the same kernels — and every later call runs that graph, dispatched as one unit instead of
  op by op from Python (the per-op eager dispatch is the cost that goes). The documented command's input never changes shape or dtype
  (the (2, 524288, 4) one-hot pair), so the function is traced once; ``tf.function`` itself would trace again for another signature and
  the stamp counts traces. TensorFlow is taken from ``sys.modules`` (the entry script imported it through baskerville before the model was
  built); a process without it loaded keeps the eager call and the stamp says so.
* the copy-free return: ``preds.numpy()`` as is, ``astype(dtype)`` only when the dtype differs (float32 -> float32 leaves the bytes as
  they are; a C-contiguous, writeable float32 array — the post never writes into it: ``untransform_preds`` rebinds before its in-place line).

``head_i`` and a non-float32 ``dtype`` (never used by the documented command) go to the stock ``__call__`` verbatim. No arithmetic
operation, operand, reduction order, precision or kernel is touched, so every value equals the stock call's; the graph holds its own
intermediate buffers on the device, so the process's device high-water mark is higher than the eager call's.

The wrapper stands in for ``seqnn_model`` in the entry script's loop (every other attribute delegated); :func:`install_class` puts the
same call on ``SeqNN.__call__`` itself, so any program that calls a ``SeqNN`` gets it (the ``BORZOI_OPT`` hook's library route).
``KIT_FWD`` outside {0, 1} raises by name."""
from __future__ import annotations

import os
import sys

FORMS = {"0": "off", "1": "graph_copy_free"}
GRAPH = "tf.function(jit_compile=False) from call 2; call 1 eager"
_STOCK_CALL = None                     # SeqNN.__call__ as shipped, kept when install_class patches the class (the head_i path calls it, never the patched one)


class ForwardCall:
    """``seqnn_model`` stand-in: the stock ``SeqNN.__call__`` (baskerville seqnn.py:1046-1060) run eagerly on the first call and as ONE
    traced graph (``tf.function``, XLA off) from the second, with the copy-free return; every other attribute delegated to the stock object."""

    def __init__(self, seqnn_model, stamp: dict):
        object.__setattr__(self, "_m", seqnn_model)
        object.__setattr__(self, "_stamp", stamp)
        object.__setattr__(self, "_graph", None)                   # (model, traced function) after the first call

    def __call__(self, x, head_i=None, dtype="float32"):
        import numpy as np
        m = object.__getattribute__(self, "_m")
        st = object.__getattribute__(self, "_stamp")
        if head_i is not None:
            st["n_stock_calls"] += 1
            call = _STOCK_CALL or type(m).__call__               # the stock call, verbatim (a head selection this form does not cover)
            return call(m, x, head_i=head_i, dtype=dtype)
        model = m.ensemble if m.ensemble is not None else m.model  # seqnn.py:1049-1054 with head_i None
        g = object.__getattribute__(self, "_graph")
        tf = sys.modules.get("tensorflow")
        if g is not None and g[0] is model:
            preds = g[1](x).numpy()                               # calls 2..: the traced graph of the same model (one dispatch), the device-to-host copy
            st["n_graph_calls"] += 1
            st["n_traces"] = int(g[1].experimental_get_tracing_count())
        else:
            preds = model(x).numpy()                              # call 1: the eager call and the device-to-host copy, as the stock (cuDNN's algorithm choice made here, as in a stock process)
            st["n_eager_calls"] += 1
            if tf is not None:                                    # trace the same model for the calls to come (TensorFlow is loaded in the entry: baskerville imported it)
                object.__setattr__(self, "_graph", (model, tf.function(model, jit_compile=False)))
                st["graph"] = GRAPH
            else:
                st["graph"] = "eager (tensorflow not loaded in this process)"
        if preds.dtype != np.dtype(dtype):
            preds = preds.astype(dtype)                           # the stock's second copy only when it changes the dtype
        else:
            st["n_copy_free"] += 1
        st["n_calls"] += 1
        return preds

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_m"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_m"), name, value)


def new_stamp(wrapped: str) -> dict:
    return {"kit_fwd": FORMS["1"], "enabled": True, "env": "KIT_FWD", "form": "call 1 eager; calls 2.. one traced graph (tf.function, XLA off); .numpy(); astype only on a dtype change",
            "graph": "armed" if "tensorflow" in sys.modules else "eager (tensorflow not loaded in this process)", "wrapped": wrapped,
            "n_calls": 0, "n_eager_calls": 0, "n_graph_calls": 0, "n_traces": 0, "n_copy_free": 0, "n_stock_calls": 0}


def install_class(SeqNN) -> dict:
    """The library route: ``SeqNN.__call__`` itself becomes the kit's forward call for every instance in this process (one ``ForwardCall``
    per instance, made on its first call; one stamp for the process, its counters summed over instances). Returns the stamp. Idempotent."""
    global _STOCK_CALL
    if getattr(SeqNN, "_kit_forward_stamp", None) is not None:
        return SeqNN._kit_forward_stamp
    stamp = new_stamp("SeqNN.__call__ (every instance)")
    _STOCK_CALL = SeqNN.__call__

    def __call__(self, x, head_i=None, dtype="float32"):
        fc = self.__dict__.get("_kit_forward")
        if fc is None:
            fc = ForwardCall(self, stamp)
            self.__dict__["_kit_forward"] = fc
        return fc(x, head_i=head_i, dtype=dtype)

    __call__.__doc__ = _STOCK_CALL.__doc__
    SeqNN.__call__ = __call__
    SeqNN._kit_forward_stamp = stamp
    return stamp


def install(seqnn_model):
    """Returns ``(model_for_the_loop, stamp)``: the stock object itself when ``KIT_FWD`` is off, else the ``ForwardCall`` wrapper."""
    v = os.environ.get("KIT_FWD", "0")                            # the literal name: the package reads the kit's switches from these bytes
    if v not in FORMS:
        raise ValueError(f"KIT_FWD={v!r}: not one of {sorted(FORMS)}")
    if v == "0":
        return seqnn_model, {"kit_fwd": "off", "enabled": False, "env": "KIT_FWD"}
    stamp = new_stamp("ensemble" if seqnn_model.ensemble is not None else "model")
    return ForwardCall(seqnn_model, stamp), stamp
