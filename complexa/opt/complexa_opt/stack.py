"""The machine and the tree as found: where the engine directory is, its pins, upstream's checkout and console script, the weights directory,
the GPU, and the (inactive) activation report of the stock mode.

Tree: ``tree_home()`` = ``$MODEL_OPT`` (run.sh's name for the ``complexa/`` directory) else two levels above this package; it must hold
``stock/PINS.json`` and ``opt/`` (``opt_core.home``). Pins: ``pins()`` = ``stock/PINS.json``; ``check_pins`` = ``stock/check_pins.py``
imported from the tree (standard library; the one reader of the pin's gates — the installed ``proteinfoundation`` version, the checkout's
commit and cleanliness under src/ and configs/, the pinned config files, the weights). Upstream: ``$LOCAL_CODE_PATH`` (upstream's own variable, ``env.sh`` / the CLI's
``init``) names the checkout the pipeline config is read from (no default: STOCK.md 'Variables'); ``complexa`` is upstream's console
script on PATH. Weights: ``$CKPT_PATH`` (upstream's own variable: ``configs/search_binder_pipeline.yaml``
``ckpt_path: ${oc.env:CKPT_PATH}``) names the directory holding ``complexa.ckpt`` and ``complexa_ae.ckpt``; the stock route passes it as
``++ckpt_path`` / ``++autoencoder_ckpt_path`` (the local pipeline config ships ``./ckpts`` relative paths). GPU: ``gpu()`` =
``opt_core.gates.nvidia_smi_probe`` (no torch in the parent); the target class check is the core's (``MODEL_OPT_TARGET_GPU``,
``gpu_class_check``) and is a report on the stock route, never a gate.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from typing import Optional

from opt_core import gates, home, jit_cache

from . import TAG, ActivationError, __version__, modes

ENV_UPSTREAM = "LOCAL_CODE_PATH"             # upstream's checkout (upstream's own variable name)
ENV_WEIGHTS = "CKPT_PATH"                    # the directory holding complexa.ckpt + complexa_ae.ckpt (upstream's own variable name)
ENV_TARGET_GPU = "MODEL_OPT_TARGET_GPU"
PACKAGE_ENV = (modes.ENV_MODE, modes.ENV_RECORD)   # this package's own variables: never present in a stock child (the kit route exports them itself)
PTH_FILE = "complexa_opt_autoload.pth"       # laid into site-packages by the install (opt/_build_backend.py): `import complexa_opt._autoload` at interpreter start
PTH_MODULES = ("complexa_opt", "complexa_opt._autoload")   # what that line loads in EVERY interpreter of the environment, a stock child included: the declared inert pair (no core, no torch, no finder unless COMPLEXA_OPT names a kit mode)
INTERPRETER_ENV = ("PYTHONSAFEPATH",)        # removed from the child and recorded: `-P` semantics differ per entry point; one rule for every entry
CONSOLE_SCRIPT = "complexa"                  # upstream's entry point (pyproject.toml [project.scripts] complexa = proteinfoundation.cli.cli_runner:main)
PIPELINE_CONFIG = os.path.join("configs", "search_binder_local_pipeline.yaml")   # the pipeline config the stock route names (relative to the checkout)

_PINS = None
_CHECK_PINS = None


def tree_home() -> str:
    try:
        return home.tree_home(__file__, require=(os.path.join("stock", "PINS.json"), "opt"))
    except home.HomeError as e:
        raise ActivationError(str(e)) from None


def stock_dir() -> str:
    return os.path.join(tree_home(), "stock")


def package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def pins() -> dict:
    global _PINS
    if _PINS is None:
        with open(os.path.join(stock_dir(), "PINS.json"), encoding="utf-8") as fh:
            _PINS = json.load(fh)
    return _PINS


def check_pins():
    """``stock/check_pins.py`` as a module (imported from the tree by path; standard library only)."""
    global _CHECK_PINS
    if _CHECK_PINS is None:
        p = os.path.join(stock_dir(), "check_pins.py")
        spec = importlib.util.spec_from_file_location("complexa_check_pins", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CHECK_PINS = mod
    return _CHECK_PINS


def sha256(path: str) -> str:
    return gates.sha256_file(path)


def upstream_home(environ=None) -> str:
    """``$LOCAL_CODE_PATH``: the checkout holding ``configs/search_binder_local_pipeline.yaml``. ActivationError when unset or not a checkout."""
    environ = os.environ if environ is None else environ
    p = environ.get(ENV_UPSTREAM)
    if not p:
        raise ActivationError(f"{ENV_UPSTREAM} is not set: it names upstream's Proteina-Complexa checkout at the pin (the directory holding {PIPELINE_CONFIG}); export it (README 'Variables')")
    p = os.path.abspath(p)
    if not os.path.isfile(os.path.join(p, PIPELINE_CONFIG)):
        raise ActivationError(f"{ENV_UPSTREAM}={p} holds no {PIPELINE_CONFIG} (not upstream's checkout)")
    return p


def pipeline_config(environ=None) -> str:
    """The absolute path of the pipeline config the stock route passes to ``complexa generate``."""
    return os.path.join(upstream_home(environ), PIPELINE_CONFIG)


def weights_dir(environ=None) -> str:
    environ = os.environ if environ is None else environ
    p = environ.get(ENV_WEIGHTS)
    if not p:
        raise ActivationError(f"{ENV_WEIGHTS} is not set: it names the directory holding {', '.join(w['file'] for w in pins()['weights']['files'])}; export it (README 'Variables')")
    return os.path.abspath(p)


def weights_gate(environ=None, *, with_sha: bool = False) -> dict:
    """``{"dir", "files", "pinned", "line", "bad", "detail"}`` of the weights directory against stock/PINS.json (check_pins.check_weights): both
    checkpoints present at their pinned byte counts; their sha256 digests are the pin's citation of those bytes, printed on the line and not
    recomputed unless ``with_sha`` (``check_pins.py --weights <dir>`` is the by-hand digest check)."""
    d = weights_dir(environ)
    cp = check_pins()
    bad, detail = cp.check_weights(pins(), d, with_sha=with_sha)
    return {"dir": d, "files": {w["file"]: w["sha256"] for w in pins()["weights"]["files"]}, "pinned": not bad, "bad": bad, "detail": detail,
            "line": cp.weights_line(pins(), d, pinned=not bad, with_sha=with_sha)}


def console_script() -> dict:
    """``{"path", "interpreter"}`` of upstream's ``complexa`` on PATH (the interpreter read from its ``#!`` line, for the record); ActivationError when absent."""
    p = shutil.which(CONSOLE_SCRIPT)
    if not p:
        raise ActivationError(f"upstream's console script {CONSOLE_SCRIPT!r} is not on PATH (pip install -e $LOCAL_CODE_PATH puts it there)")
    interp = None
    try:
        with open(p, "rb") as fh:
            head = fh.readline().decode("utf-8", "replace").strip()
        if head.startswith("#!"):
            interp = head[2:].strip().split()[0]
    except OSError:
        pass
    return {"path": p, "interpreter": interp}


def gpu() -> dict:
    """The GPU as nvidia-smi reports it (no torch import in the parent): {name, cc, sm, memory_mib, probe}."""
    try:
        return gates.nvidia_smi_probe()
    except Exception as e:                                   # a report: no GPU is recorded, the stock routes decide nothing on it
        return {"name": None, "probe": f"nvidia-smi failed: {type(e).__name__}: {e}"}


def target_gate(g: Optional[dict] = None, environ=None) -> gates.Gate:
    environ = os.environ if environ is None else environ
    return gates.gpu_class_check(g if g is not None else gpu(), environ.get(ENV_TARGET_GPU) or None)


def stack_key() -> str:
    """``torch<version>-cu<cuda>`` of the installed torch from distribution metadata (no import), e.g. ``torch2.7.0-cu126``; ``unknown`` when torch
    is absent. A PyPI torch wheel carries no local tag: its CUDA line is read from the ``nvidia-cuda-runtime-cu12`` wheel it depends on."""
    v = gates.dist_version("torch")
    if not v:
        return "unknown"
    base, _, local = v.partition("+")
    if not local:
        rt = gates.dist_version("nvidia-cuda-runtime-cu12") or ""
        parts = rt.split(".")
        local = f"cu{parts[0]}{parts[1]}" if len(parts) >= 2 and parts[0].isdigit() else "cpu"
    return f"torch{base}-{local}"


def jit_key() -> str:
    """``torch<version>-cu<cuda>-sm<cc>``, the directory the shared JIT-cache root's caches are keyed by (configs/*.env: ``MODEL_OPT_JIT_KEY``
    under ``MODEL_OPT_JIT_ROOT``): the core's one key rule (``opt_core.jit_cache.key``) — the installed torch and its CUDA line from distribution
    metadata, the card's compute capability from nvidia-smi, no torch import — e.g. ``torch2.7.0-cu126-sm90`` on an H100, ``torch2.7.0-cu126-sm80``
    on an A100. A part this box cannot establish (no torch metadata, no GPU visible to nvidia-smi) makes the whole key the word ``unknown`` —
    named on stderr in one ``JIT-CACHE key=unknown: <the core's words>`` line, never guessed and never a refusal: configs/h100.env then keys the
    caches under ``$MODEL_OPT_JIT_ROOT/unknown/``, a directory no pinned stack shares, and the run goes on."""
    try:
        return jit_cache.key()
    except jit_cache.StackKeyUnknown as e:
        sys.stderr.write(f"[{TAG}] JIT-CACHE key=unknown: {e} (the shared root's caches go under <root>/unknown/; nothing is refused over it)\n")
        return "unknown"


def stock_report(mode: str) -> dict:
    """The activation report of the stock route (inactive by definition: nothing to activate) for the manifest."""
    return {"active": False, "mode": mode, "route": modes.route_of(mode), "reason": f"mode {mode} ({modes.route_of(mode)} route): a stock route activates nothing",
            "package_version": __version__, "levers": []}


def hook_probe(env: dict, python: Optional[str] = None, cwd: Optional[str] = None) -> dict:
    """Run ``python -c`` (default: the interpreter upstream's console script runs) in ``env`` — the kit child's environment, COMPLEXA_OPT set —
    and report whether the autoload hook came up by itself: ``{python, pth, autoload_loaded, finder_armed, present, rc, stderr}``.
    Nothing of the model is imported by the probe (the finder arms at start and waits for its trigger)."""
    import subprocess
    if python is None:
        try:
            python = console_script().get("interpreter") or sys.executable
        except ActivationError:                                # no console script on PATH: this interpreter (the probe then reports what IT holds)
            python = sys.executable
    code = ("import json, os, site, sys\n"
            "m = sys.modules.get('complexa_opt._autoload')\n"
            "pk = sys.modules.get('complexa_opt')\n"
            "dirs = [d for d in (site.getsitepackages() + [site.getusersitepackages()]) if isinstance(d, str)]\n"
            f"pth = next((os.path.join(d, {PTH_FILE!r}) for d in dirs if os.path.isfile(os.path.join(d, {PTH_FILE!r}))), None)\n"
            f"print(json.dumps({{'python': sys.executable, 'pth': pth, 'autoload_loaded': m is not None, 'package': (os.path.dirname(pk.__file__) if pk is not None and getattr(pk, '__file__', None) else None), 'finder_armed': bool(m is not None and getattr(m, 'FINDER', None) is not None and getattr(m.FINDER, 'armed', False)), 'present': sorted(k for k in os.environ if k in {list(PACKAGE_ENV)!r} or k.startswith('COMPLEXA_'))}}))\n")
    r = subprocess.run([python, "-c", code], env=env, cwd=cwd, capture_output=True, text=True, timeout=120)
    out = {"python": python, "pth": None, "autoload_loaded": False, "package": None, "finder_armed": False, "present": [], "rc": r.returncode, "stderr": r.stderr.strip()[-600:]}
    if r.returncode == 0:
        try:
            out.update(json.loads(r.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            out["stderr"] = (out["stderr"] + " | stdout: " + r.stdout[-300:]).strip()
    return out
