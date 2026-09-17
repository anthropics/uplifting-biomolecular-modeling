"""TensorFlow's determinism settings ("the deterministic recipe"), composed identically on every arm that runs them: `--mode off --det 1` (CHROMBPNET_OPT_DET=1)
puts them on the stock console script's child; `--mode exact` puts the same block on the kit's own process (modes.DET_MODES); `fast` never runs them.

The kit's seed hook is `opt/kit_ho/tf/det_subprocess/sitecustomize.py`: at interpreter start, when CHROMBPNET_DET_SUBPROCESS == "1",
it seeds TensorFlow with CHROMBPNET_DET_SEED, enables op determinism, turns TF32 off and prints `[det_subprocess] applied ...` to
stderr. The package puts that directory FIRST on PYTHONPATH and sets the two variables
the file reads; the recipe's environment block (TF_ENV) and, on the K1 route, the cuBLAS workspace pin (K1_ENV) complete it. The kit's
route decision reads the mode from these variables (fastdefault.route_mode: TF_DETERMINISTIC_OPS / TF_USE_DEFAULT_CONV_ALGO /
CHROMBPNET_DET_SUBPROCESS == "1" -> det). The stock has no deterministic mode of its own (TF_DETERMINISTIC_OPS=1 alone makes TF 2.8's
load_model refuse for want of a seed — see the hook's module text): the stock arm under --det is the stock CLI plus this same recipe, and the
seed hook is what makes it deterministic. The batch stays the stock default (-bs, stock parsers.py:223).

The hook rides PYTHONPATH, so no arm is started with `python -I` (which ignores PYTHONPATH).
"""
import os
from typing import Dict, Optional

ENV_SUBPROCESS = "CHROMBPNET_DET_SUBPROCESS"                 # the gate variable the seed hook reads
ENV_SEED = "CHROMBPNET_DET_SEED"                             # the seed variable the seed hook reads
SEED = "0"                                                   # the recipe's seed
SITECUSTOMIZE_RELDIR = os.path.join("tf", "det_subprocess")  # the directory that holds the kit's sitecustomize.py
SITECUSTOMIZE = "sitecustomize.py"
TF_ENV = {"TF_DETERMINISTIC_OPS": "1", "TF_USE_DEFAULT_CONV_ALGO": "1", "TF_CUDNN_USE_AUTOTUNE": "1", "TF_XLA_FLAGS": "--tf_xla_auto_jit=0",
          "TF_CPP_MIN_LOG_LEVEL": "0", "TF_CPP_VMODULE": "gpu_utils=2"}   # type: Dict[str, str]
K1_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}                          # the K1 route only
K1_ROUTE = "k1"
NAMES = (ENV_SUBPROCESS, ENV_SEED) + tuple(TF_ENV) + tuple(K1_ENV) + ("PYTHONPATH",)   # every variable the recipe touches


def sitecustomize_dir(kit_home: str) -> str:
    return os.path.join(kit_home, SITECUSTOMIZE_RELDIR)


def requested(environ: Optional[dict] = None) -> bool:
    """CHROMBPNET_OPT_DET=1 in the environment = `--mode off --det 1` (the environment spelling of the stock arm's setting; redundant with `exact`, refused by name with `fast`)."""
    environ = os.environ if environ is None else environ
    return (environ.get("CHROMBPNET_OPT_DET") or "").strip() == "1"


def env(kit_home: str, base: Optional[dict] = None, route: Optional[str] = None) -> Dict[str, str]:
    """`base` (default: the process environment) with the recipe composed: the kit's sitecustomize dir prepended to PYTHONPATH, the two
    variables the file reads, the TF block, and the cuBLAS pin when `route` is the K1 route. Returns a new dict."""
    out = dict(os.environ if base is None else base)
    d = sitecustomize_dir(kit_home)
    if not os.path.isfile(os.path.join(d, SITECUSTOMIZE)):
        raise FileNotFoundError("the kit's seed hook is missing: {}".format(os.path.join(d, SITECUSTOMIZE)))
    prev = [p for p in (out.get("PYTHONPATH") or "").split(os.pathsep) if p and p != d]
    out["PYTHONPATH"] = os.pathsep.join([d] + prev)
    out[ENV_SUBPROCESS] = "1"
    out[ENV_SEED] = SEED
    out.update(TF_ENV)
    if route == K1_ROUTE:
        out.update(K1_ENV)
    return out


def describe(environ: dict, on: bool, route: Optional[str] = None) -> dict:
    """The recipe block of the manifest."""
    if not on:
        return {"on": False, "env": {}, "seed_hook": None}
    return {"on": True, "seed_hook": os.path.join(SITECUSTOMIZE_RELDIR, SITECUSTOMIZE), "route": route,
            "env": {k: environ.get(k) for k in NAMES if k in environ}}


def kit_hook_reads(kit_home: str) -> dict:
    """What the kit's sitecustomize.py reads from the environment (the lock test): the gate variable and the seed variable, by regex."""
    import re
    with open(os.path.join(sitecustomize_dir(kit_home), SITECUSTOMIZE), "r", encoding="utf-8") as fh:
        src = fh.read()
    names = re.findall(r"os\.environ\.get\(\"([A-Z0-9_]+)\"", src)
    return {"gate": names[0] if names else None, "seed": names[1] if len(names) > 1 else None, "applies_tf_seed": "tf.random.set_seed" in src,
            "enables_op_determinism": "enable_op_determinism" in src, "tf32_off": "enable_tensor_float_32_execution(False)" in src}
