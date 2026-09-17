"""enformer_deepmind_opt — inference optimizations for the official Enformer (the TF-Hub SavedModel deepmind/enformer/1 of deepmind-research/enformer, TensorFlow 2), one mode: ``exact``.

The kit is a drop-in: with it enabled, the released model as users run it —
``tensorflow_hub.load("https://tfhub.dev/deepmind/enformer/1").model.predict_on_batch(x)`` (or the same SavedModel through
``tf.saved_model.load``) — returns the same bytes as stock, faster. Nothing about the model's interface changes: same call, same outputs.

    ENFORMER_DEEPMIND_OPT=exact python your_script.py     # no code change (the package's .pth arms it at interpreter start)
    import enformer_deepmind_opt; enformer_deepmind_opt.enable()   # or explicitly, before the model is loaded / first called

Every line the package prints starts with ``[enformer-deepmind-opt]`` on stderr. ``enable()`` refuses by name (``NOT ACTIVE``, nothing applied,
stock untouched) when no CUDA device is visible to TensorFlow, the op library cannot load under this TensorFlow, or the kit's files are missing; a device that merely differs
from the tested ones is named on the line and served. A SavedModel whose prediction graph is not the released Enformer's runs its stock graph,
said once on stderr. ``python -m enformer_deepmind_opt check`` (``run.sh check``) is the dry
run: the same resolution, nothing applied.
"""
from ._runtime import (PREFIX, ActivationError, apply, check, disable, enable, status)   # noqa: F401

__version__ = "0.1.0"
__all__ = ["enable", "disable", "apply", "check", "status", "ActivationError", "PREFIX", "__version__"]
