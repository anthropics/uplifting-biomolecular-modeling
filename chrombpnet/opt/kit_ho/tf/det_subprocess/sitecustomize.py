"""The deterministic recipe's seed hook for a python SUBPROCESS (the stock console script's child under `--mode off --det 1`, the kit's own process under `--mode exact`, and every python
process the kit's own arm starts under the recipe).

TensorFlow 2.8 under TF_DETERMINISTIC_OPS=1 refuses keras load_model unless a seed is set ('Random ops require a seed to be set when
determinism is enabled'), and a subprocess inherits only the environment, not an in-process tf.random.set_seed(). This module rides the
child's PYTHONPATH (chrombpnet_opt.det.env puts its directory first) and applies the same recipe — seed, op determinism, TF32 off — at
interpreter start, before any op. Gated by CHROMBPNET_DET_SUBPROCESS=1; it imports nothing of the kit.

A `sitecustomize` module shadows every other one on the path, so after its own work this hook runs the NEXT `sitecustomize.py` found on
sys.path after its own directory, in a namespace of its own with that file's __file__ and no module registered (the environment's or the
site's own start-up hook keeps working under the recipe exactly as it does without it; none present = nothing to do). The hook does its work
once per process (a flag on `sys`): a foreign hook that chains back to this file finds it inert.
"""
import os
import sys as _sys

if not getattr(_sys, "_chrombpnet_det_hook", False):          # once per process: a foreign start-up hook that chains back to this file finds it inert
    _sys._chrombpnet_det_hook = True
    if os.environ.get("CHROMBPNET_DET_SUBPROCESS") == "1":
        try:
            import tensorflow as tf
            _seed = int(os.environ.get("CHROMBPNET_DET_SEED", "0"))
            tf.random.set_seed(_seed)
            _opdet = False
            if hasattr(tf.config.experimental, "enable_op_determinism"):
                tf.config.experimental.enable_op_determinism(); _opdet = True
            # the FULL recipe (a shim without TF32-off runs the DET arms with tensor cores ON — 'disable_tensor_core=0' in cuDNN's
            # filter line — which is not this recipe): TF32 OFF here, exactly as the in-process form of the recipe
            # (enable_tensor_float_32_execution(False))
            _tf32 = None
            if hasattr(tf.config.experimental, "enable_tensor_float_32_execution"):
                tf.config.experimental.enable_tensor_float_32_execution(False); _tf32 = False
            _stamp = f"seed={_seed};op_determinism={int(_opdet)};tf32={_tf32};tf={tf.__version__}"
            os.environ["CHROMBPNET_DET_SUBPROCESS_APPLIED"] = _stamp
            print(f"[det_subprocess] applied {_stamp} pid={os.getpid()}", file=_sys.stderr, flush=True)   # the stamp PRINTED per arm (the rule)
        except Exception as _e:                                   # noqa: BLE001 — a process without TF (helpers) is not an error
            os.environ["CHROMBPNET_DET_SUBPROCESS_APPLIED"] = f"skipped:{type(_e).__name__}"
    # chain: the next sitecustomize.py on sys.path after this directory, run as the interpreter would have run it (its own __file__; no module
    # is registered under a new name, so the process's module table is what it would be with this hook standing in for the displaced one)
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _p in _sys.path:
            _c = os.path.join(_p or os.getcwd(), "sitecustomize.py")
            if os.path.isfile(_c) and os.path.dirname(os.path.abspath(_c)) != _here:
                with open(_c, "rb") as _fh:
                    _src = _fh.read()
                _ns = {"__name__": "sitecustomize", "__file__": _c, "__builtins__": __builtins__}
                exec(compile(_src, _c, "exec"), _ns)            # noqa: S102 — the displaced start-up hook's own code
                break
    except Exception:                                          # noqa: BLE001 — a broken foreign hook must not break the recipe's process
        pass
