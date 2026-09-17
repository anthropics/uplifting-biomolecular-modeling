"""The modes, resolved against the kit's own files, and the gate every route runs before anything executes.

``exact`` | ``off`` are the modes this package promises:

* ``exact`` — THE DEFAULT and the one kit mode: the kit's ``v17/borzoi_sad.py`` with its one composition (``kit.KIT_COMPOSITION``:
  ``KIT_FWD=1`` — ``v17/kitlib/forward.py``: the forward traced once and run as one graph from the second call, the copy-free return) and
  its one statistics writer (``v17/kitlib/writer.py``: the stock's expressions, only the requested passes). Every lever states the stock's
  bytes by construction; no further switch.
* ``off``   — the pinned stock ``borzoi_sad.py`` in a clean subprocess (``stock_sad.py``), nothing of the kit on the path.

``--det 1`` (``sad`` / ``check``) exports :data:`DET_RECIPE` into the job — the reproducibility setting: numpy's CPU dispatch pinned to one
instruction set and one BLAS thread, so the host-side arithmetic is byte-stable across hosts, and cuDNN's convolution autotuning off, so the
forward's algorithm choice is the library's fixed heuristic rather than a per-process timing race (byte-stable process to process on every
card); off = production (TensorFlow's and numpy's defaults).

The kit is an entry-script swap (its levers live in the entry's ``main()``), so a mode is applied at the process boundary — by
``borzoi-opt sad`` (cli.py) or the ``BORZOI_OPT`` hook (_autoload.py). :func:`resolve` is the one gate: the kit tree present,
the stock entry pinned, the environment free of ANY kit switch (a KIT_* name the kit reads, set by the user, is refused by name — a mode
is one composition), the GPU read; it returns the activation report and applies nothing. :func:`enable` (the Python route) runs the
same gate and names why in-process activation is impossible.
"""
from __future__ import annotations

import importlib.metadata as _md
import importlib.machinery
import os
import platform
import shutil
import subprocess
import sys
from typing import Dict, Optional

from . import kit

MODES = ("off", "exact")
DEFAULT_MODE = "exact"                                           # the package default and the one kit mode (off is selected by name)
KIT_MODES = ("exact",)                                           # the modes the kit itself serves (off runs the stock script, no kit)
from . import ENV                                                # "BORZOI_OPT", defined once in the package __init__ (the start-up hook reads it without importing this module)
KIT_ENV_PREFIX = "KIT_"                                          # each switch the kit reads is a KIT_* variable (kitlib)
TARGET_GPU_ENV = "MODEL_OPT_TARGET_GPU"
TESTED_GPU_CC = {"9.0": "H100", "8.0": "A100"}                     # the compute capabilities the configurations cover (configs/h100.env, a100.env); any other card runs — named on the activation line, never a gate
DET_RECIPE = {                                                   # `sad --det 1`: the reproducibility pin — numpy's CPU dispatch + BLAS threads (host arithmetic byte-stable across hosts)
    "NPY_DISABLE_CPU_FEATURES": "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR,AVX2,FMA3,X86_V3,X86_V4",
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "TF_CUDNN_USE_AUTOTUNE": "0",                                 # + cuDNN autotuning off: the convolution algorithm is the library's heuristic choice, the same in every process (at
}                                                                # TensorFlow's default the choice is timed per process and can differ between processes on some cards)

_LAST: Optional[dict] = None


class ActivationError(RuntimeError):
    """A requested mode cannot run on this machine: the kit is absent or the stock is not the pinned one, the environment conflicts,
    or the route cannot apply the mode (named in the message)."""


def _dist_version(name: str) -> Optional[str]:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def package_version() -> Optional[str]:
    v = _dist_version("borzoi_opt") or _dist_version("borzoi-opt")
    if v:
        return v
    pkg = sys.modules.get("borzoi_opt")
    return getattr(pkg, "__version__", None) if pkg is not None else None


def stack_versions() -> Dict[str, Optional[str]]:
    """Installed distribution versions of the stock stack, from metadata only (nothing is imported)."""
    return {k: _dist_version(k) for k in ("tensorflow", "baskerville", "borzoi", "numpy", "h5py", "pandas", "pysam", "scipy")}


def stack_drift(versions: Dict[str, Optional[str]], root: Optional[str] = None) -> list:
    """``<name> <found> (tested <pinned>)`` for the interpreter and every stock-stack distribution in ``versions`` whose installed version
    differs from ``stock/PINS.json`` (``python`` and the ``pins`` table). A record for the activation line, never a gate: the levers are
    host-side numpy and engage on any stack the stock itself runs on."""
    try:
        p = kit.pins(root)
    except (OSError, ValueError, kit.KitError) as e:
        return [f"stock/PINS.json unreadable ({type(e).__name__}): the tested stack versions are unknown"]
    tested = dict(p.get("pins") or {})
    if p.get("python"):
        tested["python"] = str(p["python"])
    found = dict(versions or {})
    found["python"] = platform.python_version()
    out = []
    for name in ("python",) + tuple(k for k in found if k != "python"):
        want = tested.get(name)
        if want is None:
            continue
        have = found.get(name)
        if have is None:
            out.append(f"{name} not installed (tested {want})")
        elif str(have) != str(want):
            out.append(f"{name} {have} (tested {want})")
    return out


def stock_importable() -> Dict[str, bool]:
    """Whether the stock packages the entry scripts import resolve on this interpreter's path (the path finder alone: nothing is
    imported, and the hook's own finder on the stock library is not consulted)."""
    return {n: importlib.machinery.PathFinder.find_spec(n) is not None for n in ("tensorflow", "baskerville", "h5py", "pysam")}


def gpu_info() -> Optional[dict]:
    """The first GPU nvidia-smi reports: ``{"name", "memory_mib", "cc"}``; None without nvidia-smi or a GPU."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    info: dict = {"name": parts[0] if parts else None}
    if len(parts) > 1:
        try:
            info["memory_mib"] = int(float(parts[1]))
        except ValueError:
            info["memory_mib"] = parts[1]
    if len(parts) > 2:
        info["cc"] = parts[2]
    return info


def gpu_matches_target(gpu: Optional[dict], target: Optional[str]) -> Optional[bool]:
    """None when there is no target or no GPU; else whether the target word (H100 / H200) is in the GPU's name."""
    if not target or not gpu or not gpu.get("name"):
        return None
    return target.upper() in str(gpu["name"]).upper().replace(" ", "")


def kit_mode_table(root: Optional[str] = None) -> dict:
    """The kit mode resolved by name against the kit's own files: ``{"exact": {"writer": <the kit's writer name>, "env": the composition
    (kit.KIT_COMPOSITION), "tier": <the kit's tier words>}}`` (kit.writer_table reads ``v17/kitlib/writer.py`` by AST). Raises KitError
    when the kit's writer file no longer has the expected shape."""
    wt = kit.writer_table(root)
    return {"exact": {"writer": wt["writer"], "env": dict(kit.KIT_COMPOSITION), "tier": wt["tier"]}}

def check_mode(mode: Optional[str]) -> str:
    m = (mode or DEFAULT_MODE).strip().lower()
    if m not in MODES:
        raise ActivationError(f"unknown mode {mode!r} (expected {'|'.join(MODES)})")
    return m


def kit_env_overrides(environ, root: Optional[str] = None) -> Dict[str, str]:
    """The kit's own switches present in the environment (kit.kit_env_names: the KIT_* names the frozen dir's code reads). A
    KIT_-prefixed name the kit does not read is not a switch (another tool's variable) — ignored."""
    names = set(kit.kit_env_names(root))
    return {k: v for k, v in sorted(environ.items()) if k in names}


def resolve(mode: Optional[str] = None, *, environ=None, root: Optional[str] = None, route: str = "cli", dry_run: bool = False,
            read_gpu: bool = True, det: bool = False) -> dict:
    """Resolve and gate ``mode`` on this machine; apply nothing. Returns the activation report (``ok`` = the mode may run; ``reason`` names
    why not; ``env`` = what the job's environment gains: the composition, plus DET_RECIPE under ``det``). Raises only on an unknown mode.
    ``ok`` is false only when the mode cannot run as itself: the stock entry is not the pinned bytes (``off``; the hook checks the same for
    the swap), the kit's files are missing or unreadable, or a ``KIT_*`` switch set by the caller asks for another composition. Differences
    of the environment — GPU class or target, stack versions against ``stock/PINS.json``, the stock entry's state under a kit mode — are
    ``notes`` on the activation line, never a refusal."""
    environ = os.environ if environ is None else environ
    m = check_mode(mode)
    rep: dict = {"mode": m, "route": route, "dry_run": dry_run, "det": bool(det), "package_version": package_version(), "versions": stack_versions(),
                 "stock_importable": stock_importable(), "env_switch": environ.get(ENV), "ok": True, "reason": None, "notes": []}
    gpu = gpu_info() if read_gpu else None
    target = environ.get(TARGET_GPU_ENV)
    rep["gpu"] = gpu
    rep["target_gpu"] = target
    rep["gpu_matches_target"] = gpu_matches_target(gpu, target)
    if rep["gpu_matches_target"] is False:
        rep["notes"].append(f"GPU {gpu.get('name')!r} is not the configuration's target {target}")
    if gpu and gpu.get("cc") and str(gpu["cc"]) not in TESTED_GPU_CC:           # another architecture: the levers are host-side and engage unchanged — named, never a gate
        rep["notes"].append(f"GPU compute capability {gpu['cc']} ({gpu.get('name')}) is outside the tested classes "
                            f"({', '.join(cc + ' ' + n for cc, n in TESTED_GPU_CC.items())}): the levers engage unchanged")
    rep["stack_drift"] = stack_drift(rep["versions"], root)
    if rep["stack_drift"]:                                                      # version drift of the stack: named on the line, never a gate
        rep["notes"].append("stack differs from the tested pins: " + ", ".join(rep["stack_drift"]))
    stock = kit.check_stock(kit.ENTRY, root, environ)
    rep["stock"] = {k: stock.get(k) for k in ("entry", "sha256", "pinned", "reason")}
    if m == "off":
        rep["active"] = False
        rep["entry"] = stock["entry"]
        rep["kit"] = None
        rep["env"] = dict(DET_RECIPE) if det else {}
        rep["env_stripped"] = sorted(k for k in environ if k.startswith(KIT_ENV_PREFIX) or k == ENV)
        if not stock["ok"]:
            rep["ok"] = False
            rep["reason"] = stock["reason"]
        return _remember(rep)
    kv = kit.check_kit(root)
    rep["kit"] = {k: kv.get(k) for k in ("dir", "frozen", "frozen_dir", "entry", "declared_frozen_dirs", "reason")}
    rep["kit"]["name"] = kit.kit_name()
    if not kv["ok"]:
        rep.update(active=False, ok=False, reason=kv["reason"], entry=kv.get("entry"), env={})
        return _remember(rep)
    try:
        table = kit_mode_table(root)
    except (ActivationError, kit.KitError, OSError) as e:
        rep.update(active=False, ok=False, reason=str(e), entry=kv["entry"], env={})
        return _remember(rep)
    row = table[m]
    rep["entry"] = kv["entry"]
    rep["writer"] = row["writer"]
    rep["tier"] = row["tier"]
    rep["env"] = dict(row["env"])                                  # the composition: the kit's forward switch (KIT_FWD=1)
    if det:
        rep["env"].update(DET_RECIPE)
    rep["levers"] = [name for name, lv in _registry().items() if m in lv.modes]
    rep["kit_env_overrides"] = kit_env_overrides(environ)
    if rep["kit_env_overrides"]:                                   # one mode table: a kit switch set by the user is another composition, refused by name
        names = ", ".join(f"{k}={v}" for k, v in rep["kit_env_overrides"].items())
        rep.update(active=False, ok=False,
                   reason=f"kit switch(es) {names} set in the environment: mode {m} is the kit's own composition and takes no kit switch — unset them (a mode is one composition)")
        return _remember(rep)
    if not rep["stock_importable"]["baskerville"]:
        rep["notes"].append("baskerville is not importable on this interpreter: the kit entry runs inside the stock environment")
    if not stock["ok"]:
        rep["notes"].append(f"stock entry: {stock['reason']}")
    rep["active"] = not dry_run
    return _remember(rep)


def applied(rep: dict, stamp: Optional[dict]) -> dict:
    """The applied set of a kit run, judged from the kit's own stamp (``kit_stamp.json``, the entry's atexit record; the kit itself judges
    nothing): for every lever of the mode (``rep["levers"]``) the stamp's evidence of it — ``onehot.kit_onehot`` = LUT; ``forward.enabled``
    under the composition's ``KIT_FWD=1``, its ``n_stock_calls`` (a head selection sent to the stock call) and its ``graph`` (the traced
    forward from the second call: armed, and ``n_graph_calls`` = every call after the first); ``post`` (the chunked
    post's form, its writer against the mode's, ``n_fallback_stock_path`` — variants outside the declared shape class written by the
    stock path); ``pool.probe_ok`` (false = the cores probe could not run or was implausible: the pool was sized from affinity/quota —
    a named FALLBACK, outputs unaffected, never a partial).
    ``post_applies`` False is the kit's declared precondition (``kitlib/sad_post.py`` ``applies``: a targets file, strand sums, the
    length-preserving write, per-column statistics only — anything else runs the stock post path) and ``n_fallback_stock_path`` > 0 its
    per-variant one (the shape class): both are per-call STOCK ROUTING, recorded under ``routed`` (``post=stock:<reason>``; the post levers
    under ``routed_levers``) and named on one line — never a failure (the stock's own post wrote those bytes, the mode's other levers
    engaged); ``gated`` stays in the record for a precondition that stops the whole mode (none today).
    Returns ``levers_applied``, ``partial`` (the levers that fell back or left no evidence), ``partial_reasons``, ``gated``, ``gated_levers``,
    ``routed``, ``fallbacks`` (``[(lever, detail)]``: a heuristic that took its broad default — named, exit unaffected), ``evidence``; no stamp
    at all = every lever partial (the hook that never fired)."""
    levers = list(rep.get("levers") or [])
    evidence: Dict[str, str] = {}
    partial: list = []
    reasons: list = []
    gated: list = []
    gated_levers: list = []
    routed: list = []          # per-call STOCK routing events — options outside the chunked post's set, variants outside its shape class — NAMED (`post=stock:<reason>`), never a failure: the stock's own post wrote those bytes and the mode's other levers engaged (the rule: refuse only when the whole mode cannot run)
    routed_levers: list = []
    fallbacks: list = []       # (lever, detail): a tuned heuristic that took its broad default (the cores probe -> affinity/quota) — one FALLBACK line, the exit the job's own

    def fell(name: str, why: str) -> None:
        if name not in partial:
            partial.append(name); reasons.append(f"{name}: {why}")

    if not stamp:
        return {"levers_applied": [], "partial": levers, "gated": [], "gated_levers": [], "routed": [], "routed_levers": [], "fallbacks": [], "evidence": {},
                "partial_reasons": ["no kit stamp: kit_stamp.json was not written (the entry's atexit record never fired) — no lever has evidence"]}
    st = stamp
    if "kit_stamp" in levers:
        evidence["kit_stamp"] = f"kit_stamp.json written by the entry (kit={st.get('kit')})"
    if "onehot_lut" in levers:
        oh = st.get("onehot") if isinstance(st.get("onehot"), dict) else {}
        if oh.get("kit_onehot") == "LUT":
            evidence["onehot_lut"] = f"onehot.kit_onehot=LUT (replaced {oh.get('replaced')})"
        else:
            fell("onehot_lut", f"the LUT is not installed (stamp onehot={st.get('onehot')!r})")
    if "copy_free_forward" in levers:
        fw = st.get("forward") if isinstance(st.get("forward"), dict) else {}
        want = (rep.get("env") or {}).get("KIT_FWD") == "1"
        if want and not fw.get("enabled"):
            fell("copy_free_forward", f"KIT_FWD=1 requested, the stamp says kit_fwd={fw.get('kit_fwd', 'absent')}")
        elif int(fw.get("n_stock_calls") or 0) > 0:
            n_stock, n_calls = int(fw.get("n_stock_calls") or 0), int(fw.get("n_calls") or 0)
            fell("copy_free_forward", f"{n_stock} of {n_stock + n_calls} forward calls took the stock call (a head selection this form does not cover)")
        else:
            evidence["copy_free_forward"] = f"forward.kit_fwd={fw.get('kit_fwd')} n_calls={fw.get('n_calls')} n_copy_free={fw.get('n_copy_free')} n_stock_calls=0"
    if "graph_forward" in levers:
        fw = st.get("forward") if isinstance(st.get("forward"), dict) else {}
        n_calls, n_graph, n_traces = int(fw.get("n_calls") or 0), int(fw.get("n_graph_calls") or 0), int(fw.get("n_traces") or 0)
        if not fw.get("enabled") or not str(fw.get("graph") or "").startswith(("tf.function", "armed")):
            fell("graph_forward", f"the traced forward is not armed (stamp forward.graph={fw.get('graph')!r}, kit_fwd={fw.get('kit_fwd', 'absent')})")
        elif n_calls >= 2 and n_graph != n_calls - 1:            # every call after the first (the eager one) runs the traced graph
            fell("graph_forward", f"{n_graph} of {n_calls - 1} calls after the first ran the traced graph")
        elif n_traces > 1:                                          # the documented command's input has one shape: one trace
            fell("graph_forward", f"the forward was traced {n_traces} times (one signature expected)")
        else:
            evidence["graph_forward"] = f"forward.graph n_graph_calls={n_graph} n_eager_calls={fw.get('n_eager_calls')} n_traces={n_traces}"
    post_levers = [n for n in ("pipelined_chunked_post", "post_writer", "cores_probe") if n in levers]
    if post_levers:
        if not st.get("post_applies"):   # the option set is outside the chunked post's: the STOCK post path ran for every variant inside the kit entry — routed by name, exit unaffected
            routed_levers.extend(post_levers)
            routed.append("post=stock:options — " + ", ".join(post_levers) + ": the option set is outside the set the chunked post supports (kitlib/sad_post.py applies: "
                          "a targets file, strand sums, the length-preserving write, per-column statistics only); the stock post path wrote every variant, the mode's other levers engaged")
        else:
            post = st.get("post") if isinstance(st.get("post"), dict) else {}
            pool = st.get("pool") if isinstance(st.get("pool"), dict) else {}
            n_fb, n_var = int(post.get("n_fallback_stock_path") or 0), int(post.get("n_variants") or 0)
            if "pipelined_chunked_post" in levers:
                if str(post.get("kit_post") or "") != "column_chunked_threaded_pipelined":
                    fell("pipelined_chunked_post", f"the stamp's post is {post.get('kit_post')!r}, the mode's is the pipelined chunked post")
                else:
                    if n_fb > 0:   # variants outside the supported shape class (2-D, L % 16 == 0, the targets' width) were written by the STOCK path per call — routed by name; the lever engaged for the rest
                        routed.append(f"post=stock:shape — pipelined_chunked_post: {n_fb}/{n_fb + n_var} variants outside the supported shape class were written by the stock path per call")
                    evidence["pipelined_chunked_post"] = f"post.kit_post={post.get('kit_post')} n_variants={n_var} n_fallback_stock_path={n_fb}"
            if "post_writer" in levers:
                w = post.get("writer")
                w = w.get("writer") if isinstance(w, dict) else w
                if w != rep.get("writer"):
                    fell("post_writer", f"the stamp's writer is {w!r}, the mode's is {rep.get('writer')!r}")
                else:
                    evidence["post_writer"] = f"post.writer={w}"
            if "cores_probe" in levers:
                if pool.get("probe_ok"):
                    evidence["cores_probe"] = f"pool.threads={pool.get('threads')} probe_ok=true"
                elif pool:         # the probe could not run or its table was implausible: the pool came from affinity/quota — the lever's broad default, named, outputs unaffected
                    evidence["cores_probe"] = f"pool.threads={pool.get('threads')} probe_ok=false (affinity/quota)"
                    fallbacks.append(("cores_probe", f"os_count ({pool.get('probe_refusal') or 'the probe did not run'}) pool={pool.get('threads')}"))
                else:
                    fell("cores_probe", "no pool record in the stamp")
    return {"levers_applied": [n for n in levers if n in evidence], "partial": partial, "partial_reasons": reasons, "gated": gated, "gated_levers": gated_levers, "routed": routed, "routed_levers": routed_levers,
            "fallbacks": fallbacks, "evidence": evidence}


def _registry():
    from . import registry
    return registry.LEVERS


def _remember(rep: dict) -> dict:
    global _LAST
    _LAST = rep
    return rep


def status() -> dict:
    return dict(_LAST) if _LAST is not None else {"active": False, "reason": "borzoi_opt.enable() has not run in this process"}


def enable(mode: Optional[str] = None, *, strict: bool = False, root: Optional[str] = None) -> dict:
    """The Python route (``import borzoi_opt; borzoi_opt.enable()`` before or after importing ``baskerville``): the same gate as the CLI, then
    the library levers for THIS process — the kit's forward call on ``SeqNN.__call__`` and the LUT one-hot on ``dna.dna_1hot``, installed now
    if the stock library is already imported, else when it is (the hook's finder) — ``active`` True, one ACTIVE line (route=hook). The
    ``borzoi_sad.py``-specific post levers are the entry swap's (``borzoi-opt sad`` / ``BORZOI_OPT``), said in ``notes``. In the kit's own
    entry process the kit is already running (``active`` True, nothing installed twice). ``strict`` raises when the mode cannot run."""
    rep = resolve(mode, root=root, route="python")
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    if rep["mode"] == "off":
        rep["reason"] = rep["reason"] or "mode off: nothing to enable (stock runs through `borzoi-opt sad --mode off`)"
    elif rep["ok"] and main_file and kit.is_kit_entry(main_file, kit.ENTRY, root):
        rep["active"] = True
        rep["reason"] = None
        rep["notes"].append(f"this process is the kit entry {main_file}")
    elif rep["ok"]:
        from . import _autoload                                      # the library route, in-process: install now what is imported, arm the finder for the rest
        rep["levers_installed"] = _autoload.engage(rep["mode"], main_file or "python", rep)
        rep["active"] = True
        rep["reason"] = None
        rep["notes"].append(f"library levers ({', '.join(_autoload.LIBRARY_LEVERS)}) for this process; the {kit.ENTRY} post levers come with "
                            f"`borzoi-opt sad --mode {rep['mode']}` / `{ENV}={rep['mode']} {kit.ENTRY}`")
    _remember(rep)
    if strict and not rep.get("active"):
        raise ActivationError(rep["reason"] or "not active")
    return rep
