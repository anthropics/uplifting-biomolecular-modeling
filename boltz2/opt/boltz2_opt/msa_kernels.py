"""boltz2_opt.msa_kernels — the engine adapter of the shared core's fused MSA-module Triton cells (``opt_core.ops.msa_opm`` / ``opt_core.ops.msa_pwa``) onto
Boltz-2's MSA-module layers, through this tree's Boltz-2 entry points (``opt/forward/fpf_msa/boltz2.py``):

    opm  ``boltz.model.layers.outer_product_mean.OuterProductMean.forward`` -> ``fpf_msa.boltz2.outer_product_mean`` (``opt_core.ops.msa_opm.forward_mask_norm``): the
         module's LayerNorm + ``proj_a`` / ``proj_b`` + mask in one prologue kernel, then ONE Triton kernel that runs the outer-product GEMM over the MSA
         depth and applies the ``c_hidden² -> c_z`` output projection, the bias and ``/ num_mask`` in its epilogue — the ``[N, N, c_hidden²]`` outer
         product (stock: one ``[N, N, 128]`` fp32 slab per hidden chunk, written, divided, cast and projected eight times above 384 tokens) never
         reaches HBM (registry: ``fpf_opm``)
    pwa  ``boltz.model.layers.pair_averaging.PairWeightedAveraging.forward`` -> ``fpf_msa.boltz2.msa_pair_weighted_avg`` (``opt_core.ops.msa_pwa.forward_masked``): the
         LayerNorm / ``proj_m`` / ``proj_g`` / sigmoid prologue, the per-head weighted average over token columns and the gated ``proj_o`` epilogue fused;
         the per-head ``[N, N]`` softmax weights and the ``[S, N, c_h·heads]`` average are the only intermediates (registry: ``fpf_pwa``)

Numerics class of both: Tier 2 by construction — bf16 tensor-core operands rounded where stock's autocast rounds, fp32 accumulation in a tile order
that is not cuBLAS's, LayerNorm statistics in registers: never bit-identical to stock, run-to-run reproducible (no atomics, fixed tiles). The ``fast`` row
carries them (``modes._FAST_ROW``); ``exact`` never can.

Switch ``BOLTZ_FPF_MSA=<unit[,unit]>`` (``opm``, ``pwa``; the mode table sets it). Attached by ``boltz2_opt.worker_launch --attach msa`` right after
the worker's own model import (``boltz.model.models.boltz2``: both layer modules are imported by then); the classes are patched, so every instance the
model builds is served. Tile configuration: the bundle's own row per unit and card (``opm``: ``opt_core.ops.msa_opm.cfg_for`` — the card's row when the core carries one
for its compute capability (cc 8.0: the pointer-load layout ``a2``, sized for sm_80's shared memory), else ``f1`` — TMA descriptors + fused
prologue, Triton >= 3.4; on an older Triton the pointer-load layout ``p8`` of the same numerics class; ``pwa``: ``g_fo4p`` on every card);
``BOLTZ_FPF_MSA_OPM_CFG`` / ``BOLTZ_FPF_MSA_PWA_CFG`` name another entry of the unit's ``CFG_VARIANTS`` (handed to the bundle as ``FPF_OPM_CFG`` /
``FPF_PWA_CFG`` inside the worker; the kit strips ``FPF_*`` from every environment it composes). The configuration that served is RECORDED
(``msa_report.units.<unit>.cfg``, ``tma``, ``triton``, ``card_min_tokens``) — a layout switch is a named fact of the run, never silent. A card may
carry a token floor per unit (``CARD_MIN_TOKENS``; cc 8.0: ``opm`` from 385 tokens, where Boltz-2's hidden-chunked path begins and the kernel
gains): a call under it runs the stock forward BY NAME (``skipped_by_name:below_min_tokens``), decided before any launch.

Per call, ONE decision (``serve``): the bundle's entry point serves the call; a call it declines (``fpf_msa.FPFFallback``: ``rank_or_device`` — not a
rank-4 CUDA tensor; ``dims_not_pinned`` — a module whose widths are not the kernel's pinned Boltz-2 widths) takes the module's original forward,
COUNTED under the bundle's word; a kernel error takes the original forward for THIS call, counted under ``error:<Type>`` (the first two printed) —
and a GPU out-of-memory error propagates (``opt_core.oom.is_oom``: no fallback on out-of-memory). ``EXPECTED`` is empty: Boltz-2 2.2.1 builds its
``OuterProductMean`` (c_in 64, c_hidden 32, c_out 128) and ``PairWeightedAveraging`` (c_m 64, c_h 32, 8 heads) in the MSA module only
(``trunkv2.py`` MSALayer), so every call of a predict run is the kernels' pinned shape — any fallback word or error refuses the fail-closed exit gate
(``verdict``), a unit that received calls and served none refuses it too. Refused BY NAME at apply (``conflicting``): either unit beside
``BOLTZ_FPF_STACK`` (the add-on's own
in-process stack wires the same kernels through ``fpf_engines`` — a second producer); a carried file that is missing is
refused before import (``stack.check_kit_files``).

Report (``report()``, written as ``msa_report`` into the worker log by the attach hook): ``applied`` (registry names), per unit the core ledger's
census (``opt_core.counters.Ledger``: served / fallback_by / errors / shapes), its gate, the configuration facts, and ONE LEVER line per unit in the
core's grammar (``line``).

At ``n_gpu > 1`` (``BOLTZ_TP=<P>`` in the worker: the row-sharded trunk, ``boltz2_opt.rowpair``) neither unit is installed on any rank: the
row statements own both computations (``rowpair.REPLACED_LEVERS``: the outer-product-mean is ``_opm_add_`` over the core's
``msa.opm_rows_budgeted``, the pair-weighted averaging ``_pwa_update`` over ``msa.pwa_rows``) and never call the class forwards these
kernels replace. ``apply()`` then records the disposition ``replaced_by_rowpair`` per unit instead of patching (``msa_report.units.<u>.
execution``; the LEVER line reads ``state=skipped reason=replaced_by_rowpair execution=replaced_by_rowpair statement=<...>``), the gate passes by that name, and
``stack.evidence`` requires exactly that word at ``n_gpu > 1`` (an installed-but-idle unit there is a named problem). At ``n_gpu = 1`` the
line carries ``execution=executed:<calls>``.

Pre-call disposition (unit pwa): this unit launches the carried kernels on calls with ``S·N·HC < 2^31`` (``v [S, N, H*C]``; ``INT32_LIMIT``);
a call whose ``S·N·HC >= 2^31`` (e.g. an 8192-row MSA at 3762 tokens: 8192·3762·256) is NOT launched — ``precall_disposition`` names it
``int32_limit:S·N·HC=<n>>=2^31`` from the shapes (one token), the module's stock forward runs that call, and the count rides the LEVER line
(``execution=executed:<k>,skipped_by_name:int32_limit:<m>``, ``skipped_by_name={...}``) and ``msa_report.units.pwa.skipped_by_name``;
the gate accounts it (not a fallback, not an error). Below the limit nothing changes: the same ``serve`` call, the same bytes. unit opm
carries no such limit. The bound is this unit's launch policy, not a limit of the kernels: every element offset in the carried
``fpf_msa`` kernels is formed in int64.
"""
import atexit
import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional

from opt_core.oom import is_oom          # the core's one out-of-memory predicate: an out-of-memory error propagates, no fallback applied

TAG = "boltz2-opt"
SWITCH = "BOLTZ_FPF_MSA"
BUNDLE = "forward"                                                   # opt/-relative: the directory carrying the fpf_msa package (stack.kit_path resolves it)
PACKAGE = "fpf_msa"                                                   # the bundle's MSA-kernel package, imported as a package from its own directory (nothing else of the bundle enters sys.path)
PACKAGE_FILES = tuple(f"{BUNDLE}/{PACKAGE}/{f}" for f in ("__init__.py", "_compat.py", "boltz2.py"))   # the Boltz-2 entry points (boltz2.py, over opt_core.ops.msa_opm / msa_pwa) and the fallback class import (_compat), checked present before import
CORE_OPM = "opt_core.ops.msa_opm"                                       # the shared core's outer-product-mean cell (form mask_norm: Boltz-2's schema); its cfg_for / CFG_VARIANTS are the tile words
CORE_PWA = "opt_core.ops.msa_pwa"                                       # the shared core's pair-weighted-averaging cell (form masked)
UNITS: Dict[str, dict] = {   # unit token -> the registry lever it installs, the boltz module + class whose forward it replaces, the bundle entry point, the bundle's tile-configuration words
    "opm": {"lever": "fpf_opm", "module": "boltz.model.layers.outer_product_mean", "cls": "OuterProductMean", "entry": "outer_product_mean",
            "cfg_switch": "BOLTZ_FPF_MSA_OPM_CFG", "bundle_cfg_env": "FPF_OPM_CFG", "strategy": "LOCAL.boltz2.fused_outer_product_mean"},
    "pwa": {"lever": "fpf_pwa", "module": "boltz.model.layers.pair_averaging", "cls": "PairWeightedAveraging", "entry": "msa_pair_weighted_avg",
            "cfg_switch": "BOLTZ_FPF_MSA_PWA_CFG", "bundle_cfg_env": "FPF_PWA_CFG", "strategy": "LOCAL.boltz2.fused_pair_weighted_averaging"},
}
LEVERS = tuple(u["lever"] for u in UNITS.values())                    # the registry levers this adapter installs (worker_launch reads it)
EXPECTED = ()                                                          # no declared stock path: every MSA-module call of a Boltz-2 predict run is the kernels' pinned shape (see the module docstring)
FPF_STACK_SWITCH = "BOLTZ_FPF_STACK"                                   # the FPF add-on's own sitecustomize stacks (modes.OFF_IN_EVERY_MODE): inert at off|0|unset
_STATE: Dict[str, Any] = {"execution": {}, "n_gpu": 1, "ledgers": {}, "applied": [], "patched": [], "units": [], "facts": {}, "orig": {}, "classes": {}}


# ----------------------------------------------------------------------------------------------------------------- the row's words
def _env(environ):
    return os.environ if environ is None else environ


def units(environ=None) -> List[str]:
    """The units the row names (``BOLTZ_FPF_MSA``), in order, each once; ``ValueError`` naming an unknown token."""
    raw = [t.strip().lower() for t in (_env(environ).get(SWITCH) or "").split(",") if t.strip()]
    out: List[str] = []
    for t in raw:
        if t not in UNITS:
            raise ValueError(f"{SWITCH}={_env(environ).get(SWITCH)!r}: '{t}' is not a unit (the units are {'|'.join(UNITS)})")
        if t not in out:
            out.append(t)
    return out


def requested(environ=None) -> bool:
    return bool(units(environ))


def levers_of(environ=None) -> List[str]:
    """The registry levers the row's units install."""
    return [UNITS[u]["lever"] for u in units(environ)]


def conflicting(environ=None) -> List[str]:
    """The switches in the environment row that own or wire the same module forwards: each a sentence naming the lever and why (empty = none)."""
    env = _env(environ); out = []
    us = units(env)
    s = env.get(FPF_STACK_SWITCH, "").strip().lower()
    if us and s not in ("", "0", "off"):
        out.append(f"{FPF_STACK_SWITCH}={s}: the FPF add-on's own in-process stack wires the fpf_msa kernels through fpf_engines — a second producer for "
                   "the MSA-module forwards")
    return out


def carried_problems() -> List[str]:
    """The bundle package's files, checked present (stack.check_kit_files): a missing file is a sentence; empty = every file is there."""
    from . import stack
    problems, _ = stack.check_kit_files(list(PACKAGE_FILES))
    return list(problems)


def problems(environ=None) -> List[str]:
    """The pre-launch words (stack.attachment_problems; torch-free): the refusals apply() would raise for this environment row — an unknown unit
    token, conflicting switches, carried files off their pins — as sentences (empty = the attach will install). Nothing when the row names no unit."""
    env = _env(environ)
    try:
        us = units(env)
    except ValueError as e:
        return [str(e)]
    if not us:
        return []
    return list(conflicting(env)) + carried_problems()


# ----------------------------------------------------------------------------------------------------------------- the bundle package
def _load_package():
    """Import ``fpf_msa`` as a package from the carried bundle directory (its ``__init__`` + submodule search path; the bundle root itself never
    enters ``sys.path``: its other files stay unreachable by name). Idempotent; returns the ``fpf_msa.boltz2`` entry-point module."""
    from . import stack
    if PACKAGE not in sys.modules:
        pkg_dir = stack.kit_path(f"{BUNDLE}/{PACKAGE}")
        spec = importlib.util.spec_from_file_location(PACKAGE, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"{PACKAGE}: no importable package at {pkg_dir}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = mod
        spec.loader.exec_module(mod)
    import importlib as _il
    return _il.import_module(f"{PACKAGE}.boltz2")


def _cfg_facts(unit: str) -> Dict[str, Any]:
    """The tile configuration the bundle will launch for `unit` on this box, read from the bundle's own tables — recorded, never chosen here."""
    import importlib as _il
    facts: Dict[str, Any] = {"cfg_switch": os.environ.get(UNITS[unit]["cfg_switch"], "") or None}
    try:
        import triton
        facts["triton"] = triton.__version__
    except Exception as e:  # noqa: BLE001 — a fact, best effort
        facts["triton"] = f"unavailable: {type(e).__name__}"
    try:
        if unit == "opm":
            O = _il.import_module(CORE_OPM)
            cfg, source = O.cfg_for("mask_norm", 128)                  # the cell's ONE decision (override -> the card's row -> f1 with TMA -> p8), recorded here
            name = O.cfg_name(cfg) or "?"
            facts.update(tma=bool(O._tma_available()), cfg=(source if source.startswith("override:") else name if source == "default" else f"{name}({source})"),
                         tiles=dict(cfg), card_min_tokens=card_min_tokens("opm"))
        else:
            P = _il.import_module(CORE_PWA)
            name = os.environ.get("FPF_PWA_CFG", "").strip()
            facts.update(cfg=(name or "default"), tiles=dict(P.CFG_VARIANTS[name]) if name and hasattr(P, "CFG_VARIANTS") and name in P.CFG_VARIANTS else None)
    except Exception as e:  # noqa: BLE001 — the facts are evidence; the ledger census is the gate's input
        facts["cfg_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return facts


# ----------------------------------------------------------------------------------------------------------------- the per-call ladder
INT32_LIMIT = 35_000_000 * 256 + 1   # unit pwa's TESTED launch envelope on S * N * HC (the kernels' v [S, N, H*C]) = 8.96e9 = 4.17 x 2^31: the kernels form every V/W/G/O/OUT offset in int64 (opt_core.ops.msa_pwa:44-45, 93-94, 135-136, 154, 187-188, 231, 280-281, 327-328, 349, 391-392; grids (cdiv(N,BI), cdiv(S,BS)) <= 65,535 on y for S <= 16384) and were exercised against the stock module at S x N = 2048x1400, 8192x{1400,2048,3072,4096}, 16384x{1024,2048}, 12000x2900 (rows <= 34.8M: rel err 4.60e-3 / max|d| 2.93e-3 at every corner = the sub-2^31 error class, all finite). A call over the envelope runs the stock forward by name (SKIP_INT32 word kept for the census grammar). No predicate for unit opm.
SKIP_INT32 = "int32_limit"  # a pre-call disposition: a call at or over the limit runs the stock forward BY NAME, decided from its shapes before any launch
SKIP_BELOW = "below_min_tokens"   # the per-card size gate's disposition (CARD_MIN_TOKENS): a call under the card's token floor runs the stock forward BY NAME
# Per compute capability, additive (a card absent here has no floor: every call launches the kernel, as on H100): on cc 8.0 the OPM kernel's
# row (opt_core.ops.msa_opm _CFG_BY_CC a2) is x1.21 vs the stock module in the hidden-chunked regime Boltz-2 enters above 384 tokens
# (boltz const.chunk_size_threshold; trunkv2.py: chunk_size_outer_product = 4) and parity or slower below it, where stock's own unchunked
# path is fast (A100-SXM4-80GB op-level measurement: x0.98 at N=199 S=1024, x0.72 at S=1) — so below 385 tokens the stock forward serves the
# call on that card, counted under below_min_tokens. pwa: the default row is x3.0-x3.8 vs stock at every measured N with an MSA (no floor).
CARD_MIN_TOKENS: Dict[str, Dict[str, int]] = {"8.0": {"opm": 385}}


def device_cc() -> Optional[str]:
    """'M.m' of CUDA device 0 in this (worker) process; None without torch/CUDA (the CPU tests, the parent)."""
    if "cc" not in _STATE:
        try:
            import torch
            _STATE["cc"] = "%d.%d" % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            _STATE["cc"] = None
    return _STATE["cc"]


def card_min_tokens(unit: str) -> Optional[int]:
    """The unit's token floor on this card (CARD_MIN_TOKENS), else None = no floor."""
    return (CARD_MIN_TOKENS.get(device_cc() or "") or {}).get(unit)


def precall_disposition(unit: str, module, args, kwargs) -> Optional[str]:
    """``None`` = launch the kernel; else the NAME under which THIS call runs the module's stock forward instead, decided from the call's
    shapes before anything is launched (never from an exception). pwa: ``S·N·HC >= 2^31`` with ``m [B, S, N, c_m]`` the call's MSA
    input and ``HC = num_heads · c_h`` (the kernel's ``v [S, N, H*C]``) -> ``int32_limit S·N·HC=<n> (>= 2^31)``. opm: no limit carried;
    on a card with a token floor (CARD_MIN_TOKENS) a call with ``N < floor`` tokens -> ``below_min_tokens:N=<n><<floor>(cc<M.m>)``."""
    m = args[0] if args else kwargs.get("m")
    shape = getattr(m, "shape", None)
    if shape is None or len(shape) < 3:
        return None                                                   # not the pinned rank: the bundle's own checks word it
    floor = card_min_tokens(unit)
    if floor is not None and int(shape[-2]) < floor:
        return f"{SKIP_BELOW}:N={int(shape[-2])}<{floor}(cc{device_cc()})"
    if unit != "pwa":
        return None
    n = int(shape[-3]) * int(shape[-2]) * int(getattr(module, "num_heads")) * int(getattr(module, "c_h"))
    return f"{SKIP_INT32}:S·N·HC={n}>=2^31" if n >= INT32_LIMIT else None       # one token (the core's LEVER grammar: values carry no blank)


def skip_by_name(unit: str, ledger, why: str, args) -> int:
    """Count one call disposed of by ``why`` (kind = its first word) on the unit's ledger facts (``skipped_by_name`` {kind: n}, ``skipped_last``)
    — a named bucket beside served, neither fallback nor error: the gate accounts it, the LEVER line and msa_report carry it."""
    kind = why.split(":", 1)[0]
    by = dict(ledger.get("skipped_by_name") or {}); by[kind] = int(by.get(kind, 0)) + 1
    ledger.set("skipped_by_name", by); ledger.set("skipped_last", why)
    n = sum(by.values())
    if n <= 2:
        m = args[0] if args else None
        shape = "x".join(str(int(d)) for d in getattr(m, "shape", ())) or "?"
        sys.stderr.write(f"[{TAG}] {UNITS[unit]['lever']}: call #{n} skipped_by_name on {UNITS[unit]['cls']} input {shape} — {why}; this call runs the stock forward (decided before launch)\n")
    return n


def skipped(ledger) -> Dict[str, int]:
    return dict(ledger.get("skipped_by_name") or {}) if ledger is not None else {}


def execution_word(ledger) -> str:
    """``executed:<kernel-path calls>[,skipped_by_name:<kind>:<n>...]`` — the unit's execution census at n_gpu = 1."""
    w = f"executed:{ledger.fields().get('calls', 0)}"
    for kind, n in sorted(skipped(ledger).items()):
        w += f",skipped_by_name:{kind}:{n}"
    return w


def dispatch(unit: str, ledger, entry, fallback_exc, module, args, kwargs, orig):
    """The patched forward's body: the pre-call disposition first (a named skip runs ``orig()`` without launching anything), else ``serve``."""
    why = precall_disposition(unit, module, args, kwargs)
    if why is not None:
        skip_by_name(unit, ledger, why, args)
        return orig()
    return serve(unit, ledger, entry, fallback_exc, module, args, kwargs, orig)


def serve(unit: str, ledger, entry, fallback_exc, module, args, kwargs, orig):
    """ONE decision per call: the bundle entry point serves it; a call the bundle declines (``FPFFallback``) runs ``orig()`` under the bundle's
    word; an error runs ``orig()`` for THIS call under ``error:<Type>`` (the first two printed) — except a GPU out-of-memory error, which propagates;
    else the kernel's return, counted as served with the call's shape key."""
    m = args[0] if args else None
    shape = "x".join(str(int(d)) for d in getattr(m, "shape", ())) or "?"
    try:
        out = entry(module, *args, **kwargs)
    except fallback_exc as e:
        ledger.fallback(str(getattr(e, "reason", None) or (e.args[0] if e.args else type(e).__name__)))
        return orig()
    except Exception as e:  # noqa: BLE001 — the module's original forward serves this call; the census and the gate carry the error
        if is_oom(e): raise     # a GPU out-of-memory error propagates: no fallback applied
        n = ledger.error(e)
        if n <= 2:
            sys.stderr.write(f"[{TAG}] {UNITS[unit]['lever']}: kernel error #{n} on {UNITS[unit]['cls']} input {shape} ({type(e).__name__}: {str(e)[:240]}) — "
                             "this call runs the stock forward; the exit gate refuses\n")
        return orig()
    ledger.serve(f"{shape}/{str(getattr(m, 'dtype', '')).replace('torch.', '')}", first={"shape": shape, "chunk": kwargs.get("chunk_size", args[2] if unit == "opm" and len(args) > 2 else None) if unit == "opm" else kwargs.get("chunk_heads", args[3] if len(args) > 3 else None)})
    return out


# ----------------------------------------------------------------------------------------------------------------- apply / report
REPLACED = "replaced_by_rowpair"      # the n_gpu > 1 disposition word of both units (rowpair.REPLACED_LEVERS carries the statements)


def worker_n_gpu(env=None) -> int:
    """``P`` of this worker: the row-sharding switch ``rowpair.ENV_P`` (``BOLTZ_TP``, set by stack.child_env at n_gpu > 1; absent = 1)."""
    from .rowpair import ENV_P
    v = ((os.environ if env is None else env).get(ENV_P) or "").strip()
    return int(v) if v.isdigit() and int(v) > 1 else 1


def replaced_statement(unit: str) -> str:
    """The row statement that stands where the unit's kernel would (rowpair.REPLACED_LEVERS[<lever>])."""
    from .rowpair import REPLACED_LEVERS
    return REPLACED_LEVERS[UNITS[unit]["lever"]]


def dispositions() -> Dict[str, str]:
    """Registry levers this process disposed of BY NAME instead of installing ({lever: word}; n_gpu > 1: replaced_by_rowpair) — the launcher's
    attach hook accepts an empty apply() only against this (worker_launch._attach)."""
    return {UNITS[u]["lever"]: w for u, w in (_STATE.get("execution") or {}).items() if w == REPLACED}


def apply(spec: Optional[str] = None) -> List[str]:
    """Patch the named units' class forwards (idempotent). ``spec`` overrides the env switch (a unit list). Raises ``RuntimeError`` naming the
    refusal on conflicting switches, an unknown unit token, or carried files off their pins; returns the registry levers installed."""
    if _STATE["ledgers"] or _STATE["execution"]:
        return list(_STATE["applied"])
    env = dict(os.environ)
    if spec is not None:
        env[SWITCH] = str(spec)
    us = units(env)
    if not us:
        return []
    ngpu = worker_n_gpu(env); _STATE["n_gpu"] = ngpu
    if ngpu > 1:                                                        # the row-sharded trunk owns both computations on every rank: nothing is installed, the disposition is named
        _STATE["units"] = list(us); _STATE["execution"] = {u: REPLACED for u in us}
        if not _STATE.get("atexit"):
            atexit.register(_exit_lines); _STATE["atexit"] = True
        for u in us:
            sys.stderr.write(line(u) + "\n")                             # state=replaced_by_rowpair, the statement named
        return []
    clash = conflicting(env)
    if clash:
        raise RuntimeError(f"{SWITCH}={env.get(SWITCH)}: refused — conflicting levers: " + "; ".join(clash))
    bad = carried_problems()
    if bad:
        raise RuntimeError(f"{SWITCH}: refused — the carried bundle is not the pinned bytes: " + "; ".join(bad))
    for u in us:                                                     # the bundle reads its configuration words from FPF_* names; the kit's own switches map onto them inside this worker only
        v = (os.environ.get(UNITS[u]["cfg_switch"]) or "").strip()
        if v:
            os.environ[UNITS[u]["bundle_cfg_env"]] = v
    import importlib as _il
    from opt_core.counters import Ledger
    B2 = _load_package()
    FPFFallback = _il.import_module(f"{PACKAGE}._compat").FPFFallback
    applied: List[str] = []
    for u in us:
        U = UNITS[u]
        M = _il.import_module(U["module"]); cls = getattr(M, U["cls"]); entry = getattr(B2, U["entry"])
        if u == "opm":
            cfg = (os.environ.get("FPF_OPM_CFG") or "").strip()
            O = _il.import_module(CORE_OPM)
            if cfg and cfg not in getattr(O, "CFG_VARIANTS", {}):
                raise RuntimeError(f"{U['cfg_switch']}={cfg!r}: not a configuration of {CORE_OPM}.CFG_VARIANTS")
        else:
            cfg = (os.environ.get("FPF_PWA_CFG") or "").strip()
            P = _il.import_module(CORE_PWA)
            if cfg and cfg not in getattr(P, "CFG_VARIANTS", {}):
                raise RuntimeError(f"{U['cfg_switch']}={cfg!r}: not a configuration of {CORE_PWA}.CFG_VARIANTS")
        ledger = Ledger(U["lever"], impl=f"{PACKAGE}.boltz2:{U['entry']}", origin="core", min_tokens=None, expected=EXPECTED)
        facts = _cfg_facts(u)
        for k, v in facts.items():
            ledger.set(k, v)
        orig_forward = cls.forward

        def forward(self, *args, _u=u, _ledger=ledger, _entry=entry, _orig=orig_forward, **kwargs):
            return dispatch(_u, _ledger, _entry, FPFFallback, self, args, kwargs, orig=lambda: _orig(self, *args, **kwargs))

        cls.forward = forward
        _STATE["ledgers"][u] = ledger; _STATE["orig"][u] = orig_forward; _STATE["classes"][u] = cls; _STATE["facts"][u] = facts
        _STATE["patched"].append(f"{U['cls']}.forward"); applied.append(U["lever"])
    _STATE["applied"] = applied; _STATE["units"] = us; _STATE["execution"] = {u: "executed" for u in us}
    if not _STATE.get("atexit"):
        atexit.register(_exit_lines)                                 # the census as the process ends; once per process
        _STATE["atexit"] = True
    for u in us:
        sys.stderr.write(line(u, "on") + "\n")                       # installed: zero counters, the configuration facts
    return list(applied)


def _exit_lines() -> None:
    for u in _STATE.get("units") or []:
        sys.stderr.write(line(u) + "\n")


def undo() -> None:
    """Restore the stock forwards (tests; a worker never undoes)."""
    for u, cls in (_STATE.get("classes") or {}).items():
        cls.forward = _STATE["orig"][u]
    _STATE.update(ledgers={}, applied=[], patched=[], units=[], facts={}, orig={}, classes={}, execution={}, n_gpu=1)


def verdict(ledger) -> Dict[str, Any]:
    """The ledger's fail-closed gate: refused on an error, on any fallback (EXPECTED is empty), on calls that arrived with none served; calls
    ``skipped_by_name`` (the pre-call int32 disposition) are accounted — neither fallback nor error, never a refusal."""
    g = ledger.gate()
    return {"ok": bool(g.ok), "idle": False, "reason": g.reason}


def line(unit: str, state: Optional[str] = None, reason: Optional[str] = None) -> str:
    """One unit's activation-evidence line (the core's LEVER grammar; zero counters before the first call)."""
    U = UNITS[unit]
    ledger = _STATE["ledgers"].get(unit)
    if ledger is None:
        from opt_core.report import lever_line
        if _STATE["execution"].get(unit) == REPLACED:                # n_gpu > 1: not installed on this rank — state=skipped, the reason and the execution word name the disposition, the statement beside
            return lever_line(TAG, U["lever"], "skipped", ("impl", f"{PACKAGE}.boltz2:{U['entry']}"), ("origin", "kit"), ("n_gpu", _STATE["n_gpu"]), ("execution", REPLACED),
                              ("statement", replaced_statement(unit).split(" (")[0].replace(" ", "_")), reason=reason or REPLACED, strategy=U["strategy"])
        return lever_line(TAG, U["lever"], "off", ("impl", f"{PACKAGE}.boltz2:{U['entry']}"), ("origin", "kit"), reason=reason or "not_requested", strategy=U["strategy"])
    v = verdict(ledger)
    return ledger.line(TAG, state, reason, strategy=U["strategy"], execution=execution_word(ledger), gate="ok" if v["ok"] else "refused")


def report() -> Dict[str, Any]:
    if not _STATE["ledgers"] and not _STATE["execution"]:
        return {"applied": [], "disabled": {}, "units": {}, "lines": [], "gate": None, "patched": [], "n_gpu": _STATE["n_gpu"]}
    if not _STATE["ledgers"]:                                        # n_gpu > 1: every requested unit replaced by the row statements — named per unit, the gate passes by that name
        per = {u: {"lever": UNITS[u]["lever"], "execution": REPLACED, "statement": replaced_statement(u), "census": {"calls": 0, "served": 0, "fallback": 0, "fallback_by": {}, "errors": {}},
                   "gate": {"ok": True, "idle": True, "reason": REPLACED}, "line": line(u)} for u in _STATE["units"]}
        return {"applied": [], "disabled": {}, "units": per, "lines": [p["line"] for p in per.values()], "gate": {"ok": True, "idle": True, "reason": REPLACED},
                "patched": [], "variant": ",".join(_STATE["units"]), "expected": list(EXPECTED), "n_gpu": _STATE["n_gpu"], "execution": dict(_STATE["execution"])}
    per = {}
    for u, ledger in _STATE["ledgers"].items():
        per[u] = {"lever": UNITS[u]["lever"], "execution": execution_word(ledger), "skipped_by_name": skipped(ledger), "census": ledger.fields(), "gate": verdict(ledger), "line": line(u), **{k: v for k, v in (_STATE["facts"].get(u) or {}).items()}}
    ok = all(p["gate"]["ok"] for p in per.values())
    return {"applied": list(_STATE["applied"]), "disabled": {}, "units": per, "lines": [p["line"] for p in per.values()],
            "gate": {"ok": ok, "idle": False, "reason": None if ok else "; ".join(f"{u}: {p['gate']['reason']}" for u, p in per.items() if not p["gate"]["ok"])},
            "patched": list(_STATE["patched"]), "variant": ",".join(_STATE["units"]), "expected": list(EXPECTED), "n_gpu": _STATE["n_gpu"], "execution": {u: p["execution"] for u, p in per.items()}}
