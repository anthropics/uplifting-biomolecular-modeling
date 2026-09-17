"""The engine ADAPTER of the memory mode (D40): every memory INSTALL of this kit lives here — `apply(model, res, samples, out_dir)` runs the
resolved line's install on the loaded model before the kit server's configure() (the XL add-on's levers for `big`), and `report()` returns the
memory line's state for status(). opt_core.mem holds primitives only; nothing engine-specific leaves this module. One attach name per
composition (`--mode big`). The `--n_gpu P` resource axis is `tp.py`; the n_gpu > 1 row-sharded install is `rowpair.install_rank` on each rank
process, so every install below is the single-GPU line's; apply(n_gpu != 1) refuses by name.
"""
import importlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

from . import ActivationError, __version__
from . import report as _report
from .modes import Resolution
from .registry import LEVERS
from .stack import MODEL_MODULE, _install_sys_path, driver_dirs, kit_home            # stack's own names (stack imports this module LAST, so they exist in either import order)


XL_RELPATH = "EF2_XL_ADDON_v1"                                             # opt/forward/EF2_XL_ADDON_v1: the XL memory add-on (ef2_xl.py), beside the kit


ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"


ALLOC_POLICY = "expandable"                                                 # the memory lines' allocator policy (opt_core.mem.torch_alloc POLICIES): expandable segments — the add-on's "ES"
ALLOC_CONF_XL = "expandable_segments:True"                                # its PYTORCH_CUDA_ALLOC_CONF value (== torch_alloc.conf_for(ALLOC_POLICY); the package test holds the equality)


def alloc_export(environ=None) -> dict:
    """Export the memory lines' allocator policy through the shared primitive (opt_core.mem.torch_alloc.export) BEFORE any CUDA work of this
    process: sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True when unset, accepts it when already exactly that, and REFUSES by name
    (ActivationError 'es: ...') when the environment names another allocator configuration or torch's CUDA allocator is already initialised
    in this process (the setting is read at the first CUDA allocation: a late export would be a silent no-op). The line captures no CUDA
    graph (modes.BIG_GRAPH_CAPTURE: EF2_GRAPH_CAPTURE=0); the export call keeps the form the graph lines use (graphs_on: the policy composes
    with capture either way; stack.graph_lines_alloc_export).
    Returns the evidence facts {alloc, alloc_conf, alloc_export=exported|present} recorded in the activation report."""
    from opt_core.mem import MemLeverRefused, torch_alloc
    try:
        return dict(torch_alloc.export(ALLOC_POLICY, environ, lever="es", graphs_on=True, allow_with_graphs=True))
    except MemLeverRefused as e:
        raise ActivationError(str(e)) from e


def xl_home(kit: Optional[str] = None) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(kit or kit_home()), XL_RELPATH))


def xl_knobs(samples: int, has_msa_encoder: bool, xl_set: Sequence[str]) -> dict:
    """The composition's XL lever set (modes.KitMode.xl_set: XL_FAST_SET on the fast line) as `ef2_xl.apply`
    keywords for this model and fold call: x2b (FREE: z / relpos storage) only at one diffusion sample (a second sample would read the freed storage — the
    add-on's own limit); x4 (ESMC-6B streamed through the LM pass from the host) when the set names it, its token threshold from the line's
    fixed keywords (modes.XL_BIG_KNOBS esmc_min_tok); a lever outside the set is passed off by name."""
    on = set(xl_set)
    return {"own": False, "cond": "lean" if "x3" in on else "off",                # the add-on's loop re-issue (loopfree / own) is never applied by this kit: ef2_opt's static loop
            "lmpair": "x6" in on, "relpos": "x7" in on, "initlean": "x8" in on, "distocpu": "x10" in on,   # releases the recycle's dead pair temporaries in every mode
            "free": ("z,relpos" if ("x2b" in on and int(samples) == 1) else ""),      # x2b: the confidence head's z / relpos storage (its lm_z part rode the loop re-issue)
            "esmc_offload": "x4" in on, "esmc_min_tok": 0}


def xl_levers_on(knobs: dict) -> List[str]:
    """Registry names of the XL levers the knob set turns on (the record's `xl.levers`)."""
    on = []
    if knobs.get("free"):
        on.append("x2b")
    if knobs.get("cond") == "lean":
        on.append("x3")
    on += [n for n, k in (("x6", "lmpair"), ("x7", "relpos"), ("x8", "initlean")) if knobs.get(k)]
    if knobs.get("distocpu"):
        on.append("x10")
    if knobs.get("esmc_offload"):
        on.append("x4")
    return on


XL_STATE: dict = {"installed": False}                                       # the add-on's record + knobs after xl_install (status(), the gate)


def xl_install(model, res: Resolution, samples: int) -> dict:
    """Install the XL add-on's EXACT lever set on `model` (ef2_xl.apply) BEFORE the kit's configure(): a missing source anchor in the pinned
    upstream (the add-on re-issues stock methods from their source) raises from the add-on by name and is refused here, never skipped."""
    home = xl_home()
    if not os.path.isfile(os.path.join(home, "ef2_xl.py")):
        raise ActivationError(f"xl: {os.path.join(home, 'ef2_xl.py')} is not in the tree")
    _install_sys_path([home])
    ef2_xl = importlib.import_module("ef2_xl")
    has_msa = getattr(model, "msa_encoder", None) is not None
    knobs = x4_threshold(dict(xl_knobs(samples, has_msa, res.xl_set), **res.xl_knobs))   # the line's fixed keywords win; EF2_X4_MIN_TOKENS tunes x4's engagement (NOTE line) (recorded in the ledger)
    try:
        probes = xl_stock_probes(res.xl_set)                                         # the passthrough census probes, before the add-on captures its stock callables
        if knobs.get("esmc_offload"):
            XL_LM_TOKENS.clear(); xl_lm_probe(model)                                  # x4's size census: the LM pass's token count per call, before the add-on's hook captures the callable
        led = ef2_xl.apply(model, **knobs)
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                                 # a GPU out-of-memory error propagates; it is never re-worded into a refusal
        raise ActivationError(f"xl: ef2_xl.apply refused: {e!r}") from e
    rec = {"installed": True, "version": getattr(ef2_xl, "__version__", None), "knobs": knobs, "levers": xl_levers_on(knobs),
           "x2b": "on" if knobs["free"] else f"off (num_diffusion_samples={int(samples)}: FREE is single-sample only)",
           "x4": f"on (inputs >= {knobs['esmc_min_tok']} tokens)" if knobs["esmc_offload"] else "off (not in this line's set)", "line": res.line, "xl_set": list(res.xl_set),
           "patch_shas": led.get("patch_shas"), "ledger": led, "alloc_conf": os.environ.get(ALLOC_CONF),
           "stock_probes": probes, "samples": int(samples)}
    XL_STATE.clear(); XL_STATE.update(rec)
    return dict(rec)


def xl_stats() -> Optional[dict]:
    m = sys.modules.get("ef2_xl")
    return dict(m.stats()) if m is not None else None


XL_STOCK_CALLS: Dict[str, int] = {}                                         # calls that reached a STOCK callable the add-on had patched (its passthrough branches), by probe name


XL_PROBES = {                                                                # probe -> (upstream module, class, attribute, the XL lever whose passthrough it counts)
    "relpos": ("transformers.models.esmfold2.modeling_esmfold2_common", "ResIdxAsymIdSymIdEntityIdEncoding", "forward", "x7"),
    "init": (MODEL_MODULE, "ESMFold2Model", "_init_pair_state", "x8"),
}


XL_LM_TOKENS: List[int] = []                                                # the LM pass's token count per call (x4's engagement input), recorded by xl_lm_probe


def xl_lm_probe(model) -> bool:
    """Before ef2_xl.apply(): wrap the model class's `_compute_lm_hidden_states` (the callable the add-on's x4 hook captures and calls through)
    with a probe recording each call's token count (`input_ids.shape[-1]`, the quantity x4 compares with its threshold) in XL_LM_TOKENS, so the
    gate accounts x4 by size: engaged on every LM pass at or past the threshold, off-by-size below it. Numerics untouched."""
    cls = type(model)
    orig = getattr(cls, "_compute_lm_hidden_states", None)
    if orig is None:
        return False
    if getattr(orig, "_ef2_opt_probe", None):
        return True

    def probed(self, input_ids, *a, **k):
        XL_LM_TOKENS.append(int(input_ids.shape[-1]) if hasattr(input_ids, "shape") else 0)
        return orig(self, input_ids, *a, **k)
    probed._ef2_opt_probe = "lm_tokens"; probed.__wrapped__ = orig
    cls._compute_lm_hidden_states = probed
    return True


ENV_X4_MIN_TOKENS = "EF2_X4_MIN_TOKENS"                                           # the user's x4 engagement threshold (tokens); default modes.X4_MIN_TOKENS
X4_NOTES = {"threshold_set": 0}


def x4_threshold(knobs: dict, environ=None) -> dict:
    """Apply the user's ``EF2_X4_MIN_TOKENS`` (a non-negative integer token count) to the x4 knob (``esmc_min_tok``) when x4 is in the line's set:
    one NOTE line names the value in force; unset keeps the line's default (modes.X4_MIN_TOKENS). Decided up front; nothing about x4's engagement
    is a refusal — below the threshold the census says ``off-by-size`` and the fold proceeds. A value that is not a non-negative integer is
    refused by name (an undefined request)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(ENV_X4_MIN_TOKENS)
    if raw in (None, "") or not knobs.get("esmc_offload"):
        return knobs
    try:
        n = int(str(raw).strip())
        if n < 0:
            raise ValueError
    except ValueError:
        raise ActivationError(f"{ENV_X4_MIN_TOKENS}={raw!r}: a non-negative integer token count is required (default {knobs.get('esmc_min_tok')})") from None
    out = dict(knobs, esmc_min_tok=n)
    X4_NOTES["threshold_set"] += 1
    sys.stderr.write(f"[esmfold2-opt] NOTE x4 threshold {ENV_X4_MIN_TOKENS}={n} (line default {knobs.get('esmc_min_tok')}): the ESMC-6B host offload engages at inputs >= {n} tokens; "
                     "below it the census says off-by-size and the fold proceeds\n"); sys.stderr.flush()
    return out


def x4_census(knobs: dict, st: dict, tokens: Optional[List[int]] = None) -> dict:
    """x4's account by size: `expected` = LM passes at or past the threshold (each offloads ESMC-6B once), `events` = the add-on's
    esmc_offload_events; state `engaged` (events == expected > 0), `off-by-size` (no pass reached the threshold, no event), else `MISMATCH`."""
    thr = int(knobs.get("esmc_min_tok") or 0); ev = int(st.get("esmc_offload_events") or 0)
    seen = list(XL_LM_TOKENS if tokens is None else tokens)
    exp = sum(1 for n in seen if n >= thr)
    state = "engaged" if (exp and ev == exp) else ("off-by-size" if (exp == 0 and ev == 0) else "MISMATCH")
    return {"state": state, "threshold": thr, "lm_passes": len(seen), "max_tokens": max(seen) if seen else 0, "events": ev, "expected": exp}


def xl_stock_probes(xl_set) -> List[str]:
    """Before ef2_xl.apply(): wrap the stock callables the add-on will capture as its passthrough targets with a counting probe (the add-on
    captures `_ORIG[...]` at apply time, so every passthrough branch inside the carried bytes lands on the probe and is COUNTED; the XL
    paths re-issue their own statements and never touch it). Numerics untouched: the probe calls the stock callable and nothing else."""
    installed = []
    for key, (modname, cls, attr, lever) in XL_PROBES.items():
        if lever not in xl_set:
            continue
        owner = getattr(importlib.import_module(modname), cls)
        orig = getattr(owner, attr)
        if getattr(orig, "_ef2_opt_probe", None):
            XL_STOCK_CALLS[key] = 0; installed.append(key)
            continue

        def make(orig, key):
            def probed(*a, **k):
                XL_STOCK_CALLS[key] = XL_STOCK_CALLS.get(key, 0) + 1
                return orig(*a, **k)
            probed._ef2_opt_probe = key
            probed.__wrapped__ = orig
            return probed
        setattr(owner, attr, make(orig, key)); XL_STOCK_CALLS[key] = 0; installed.append(key)
    return installed


def xl_gate(expected_folds: Optional[int] = None, replaced: Sequence[str] = ()) -> dict:
    """After a run on an xl mode — the carry's fallback census (every passthrough / early-return branch inside the carried bytes is a
    NAMED event here): every lever the record names engaged (its ef2_xl counter), every stock probe holds exactly the calls the add-on's
    scope rule routes by construction (0), the per-fold counters equal the fold count (loop / init / distogram-to-host / FREE = 3 storages
    per fold), the row-wise levers report xl == calls. Returns the verdict record; raises ActivationError
    naming every failed check (the kit's PARTIAL line + exit). ``replaced``: the levers whose statements the n_gpu > 1 row-sharded line
    re-issues on rows (``rowpair.XL_LEVERS_REPLACED``: their wrappers are never reached under that line) — named in the verdict
    (``replaced_by_rowpair``), not counted."""
    if not XL_STATE.get("installed"):
        return {"checked": False}
    st = xl_stats() or {}
    knobs = XL_STATE.get("knobs") or {}
    replaced = [n for n in (XL_STATE.get("levers") or []) if n in set(replaced)]
    levers = [n for n in (XL_STATE.get("levers") or []) if n not in set(replaced)]
    bad = []
    x4 = None
    for name in levers:
        kind, key = LEVERS[name].probe
        if name == "x4":                                                         # engages by size: accounted against the LM passes at or past its threshold, not by presence
            x4 = x4_census(knobs, st)
            if x4["state"] == "MISMATCH":
                bad.append(f"x4 esmc_offload_events={x4['events']} expected={x4['expected']} ({x4['expected']} of {x4['lm_passes']} LM passes at >= {x4['threshold']} tokens; max {x4['max_tokens']})")
            continue
        if not st.get(key):
            bad.append(f"{name} ({key}={st.get(key)})")
    folds = expected_folds
    probes = dict(XL_STOCK_CALLS)
    for key, n in probes.items():
        if n != 0:
            bad.append(f"{key} stock calls={n} expected=0 (the add-on's passthrough branch ran outside its scope rule)")
    for name, calls, xl in (("x3", "cond_calls", "cond_xl_calls"), ("x6", "s2p_calls", "s2p_xl_calls"), ("x7", "relpos_calls", "relpos_xl_calls")):
        if name in levers and st.get(calls, 0) != st.get(xl, 0):
            bad.append(f"{name} passthrough ({st.get(calls, 0) - st.get(xl, 0)} of {st.get(calls, 0)} calls took the stock statement)")
    if folds is not None:
        for name, key, per in (("x8", "init_xl_calls", 1), ("x10", "disto_cpu_events", 1), ("x2b", "free_events", 3)):
            if name in levers and st.get(key, 0) != per * folds:
                bad.append(f"{name} {key}={st.get(key, 0)} expected={per * folds} ({per} per fold, {folds} folds)")
    rec = {"checked": True, "ok": not bad, "failed": bad, "stats": st, "stock_probes": probes, "expected_folds": folds, "replaced_by_rowpair": replaced, "x4": x4}
    XL_STATE["gate"] = rec
    if bad:
        raise ActivationError("xl gate: " + "; ".join(bad))
    return rec


def apply(model, res: Resolution, samples: int, out_dir: Optional[str] = None, n_gpu: int = 1) -> dict:
    """The memory line's install on `model` (before configure()): the XL add-on for an xl line. Returns
    the records to merge into the activation report ({"xl": rec}; {} for a line with no memory lever); raises ActivationError
    by name when an install refuses. `n_gpu` != 1 is refused by name with the axis's sentence (tp.refusal: no sharded install ships)."""
    if n_gpu != 1:
        from . import tp
        why, _code = tp.refusal(res.mode, int(n_gpu))
        raise ActivationError(f"big.apply(n_gpu={n_gpu}): " + (why or "the memory line's installs are the single-GPU line's; the n_gpu > 1 row-sharded "
                                                                  "install is rowpair.install_rank on each rank process (run.sh pred --mode big --n_gpu P launches them)"))
    out: dict = {}
    if res.composition.get("xl"):
        out["xl"] = xl_install(model, res, samples)
    return out


def report() -> dict:
    """The memory line's state for status(): the XL record + stats when installed, {} otherwise."""
    out: dict = {}
    if XL_STATE.get("installed"):
        out["xl"] = dict(XL_STATE, stats=xl_stats())
    return out
