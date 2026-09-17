"""The attention-with-pair-bias and MSA-module levers installed on the model the stock runner builds.

* ``dit_attn``, ``dit_attn_fp16``, ``atom_attn``, ``pf_attn`` -- bound through the shared core's provider
  ``opt_core.kernels.apb`` BY THE MODE'S TIER WORD (exact | fast | big) on every card by ``apb_core.py`` (its docstring is the contract:
  sites, call classes, the provider's stock-family rows served through the face where the table names them, Refusal -> the statement BY NAME,
  counted).  No kit kernel, cell table, tile table or size floor for these levers: the provider's measured table decides per call class.
  ``dit_attn_fp16`` rides ``dit_attn``'s install (fp16-operand arms admissible to the tier word); a precision lever without its kernel lever is
  refused by name (``check_requires``).  Rows: modes.README_ROWS (fast / big of cc 9.0 and 8.0; pf_attn fast of cc 9.0, modes.BIG_DROPPED
  on the big line).  Under the multi-GPU line the DiffusionTransformer runs the line's own row-sharded block functions, which do not call
  ``Attention.forward``: ``dit_attn`` is installed and idle there (``tp.TALLY`` words it inert by count); ``atom_attn`` serves every rank.
* ``opm_fused`` / ``pwa_fused`` -- the MSA module's 4 OuterProductMean / 3 MSAPairWeightedAveraging modules (the carried ``fpf_msa`` package,
  ``third_party/protenix_fpf_msa``): fused LayerNorm + projection passes around the unchanged cuBLAS einsum / one bmm, epilogues in stock's
  rounding chain (fp32 accumulation); fast + big rows of cc 9.0; a module another lever already owns is left alone and NAMED;
  ``layernorm_fast``'s kernel is not called on these 7 modules (a declared census interaction).

Where they install: the levers replace bound methods on module INSTANCES of the model the stock runner builds, so they install where that
model exists -- ``runner_seam`` (the shared core's ``patch_attr_at_import`` on ``runner.inference.InferenceRunner.__init__``), after
sampler_fuse's installer.  A lever that is switched on and cannot install names itself on the kit's stream (``DIT_ATTN:unavailable(...)``,
``ATOM_ATTN:unavailable(...)``, ``PF_ATTN:unavailable(...)``, ``OPM_FUSED:unavailable(...)``, ``PWA_FUSED:unavailable(...)``) and the error PROPAGATES -- the run stops (never a silent stock run).

Stream lines (stderr, the kit's prefix): ``DIT_ATTN:on(<n> modules; cell=<cc> <k=v ...>; opt_core.kernels.apb <core version>[; NAMED <text>])``,
``DIT_ATTN_FP16:on|off``, ``ATOM_ATTN:on(...)``, ``PF_ATTN:on(...)``, ``OPM_FUSED:on(...; fpf_msa <version>)``, ``PWA_FUSED:on(...)``; a failed
install: ``<MARK>:unavailable(<repr>)`` -- then the error propagates.

Dtype (``dtype_gate``): ``pf_attn``, ``opm_fused`` and ``pwa_fused`` serve the bf16-autocast statement only. Under stock's
``--dtype fp32`` (no autocast) or ``--dtype fp16`` every call of theirs steps aside BY NAME to the module's own statement (the class's method as
the mode has it), counted: the kit re-wraps the callable the install put on each module (``DTYPE_GATED``); the LEVER line then carries
``aside=dtype_fp32:<n>`` (``dtype_fp16``), ``calls`` counts what was served, and the verdict reads the tally (``gate_census``).

Cost: no device memory beyond the kernels' outputs; the first call per (row, dtype, head width) of a process compiles the provider row's Triton
variant into the box's Triton cache (inside the sampler's eager warm-up step, before graph capture).
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import _core  # noqa: F401  (makes the pinned opt_core importable on the no-install route)
from .report import PREFIX, TAG, _token  # noqa: F401

LEVERS: Tuple[str, ...] = ("dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused")   # registry names, in install order (the precision lever rides dit_attn's install)
KERNEL_LEVERS: Tuple[str, ...] = ("dit_attn", "atom_attn", "pf_attn", "opm_fused", "pwa_fused")            # the levers with an install (a runner-seam installer) of their own
PRECISION: Dict[str, str] = {"dit_attn_fp16": "dit_attn"}                  # precision lever -> the kernel lever whose operands it selects (registry requires)
ENVS: Dict[str, str] = {"dit_attn": "PTX_DIT_ATTN", "dit_attn_fp16": "PTX_DIT_ATTN_FP16", "atom_attn": "PTX_ATOM_ATTN", "pf_attn": "PTX_PF_ATTN",
                        "opm_fused": "PTX_OPM_FUSED", "pwa_fused": "PTX_PWA_FUSED"}
MARKS: Dict[str, str] = {"dit_attn": "DIT_ATTN:", "dit_attn_fp16": "DIT_ATTN_FP16:", "atom_attn": "ATOM_ATTN:", "pf_attn": "PF_ATTN:",
                         "opm_fused": "OPM_FUSED:", "pwa_fused": "PWA_FUSED:"}      # the marker families (stack.MARKERS; the packages' grammar)
INSTALL_FNS: Dict[str, str] = {"dit_attn": "install_dit_attn", "atom_attn": "install_atom_attn", "pf_attn": "install_pf_attn",
                               "opm_fused": "install_opm_fused", "pwa_fused": "install_pwa_fused"}   # the packages' install functions
PACKAGES: Dict[str, Tuple[str, str]] = {"core": ("protenix_opt.apb_core", "opt_core.kernels.apb"), "msa": ("protenix_fpf_msa", "fpf_msa")}   # key -> (import name, the name its stream lines print): "core" = the kit's tier-word binding of the shared core's provider; "msa" = the carried package under third_party/
LEVER_PKG: Dict[str, str] = {"dit_attn": "core", "dit_attn_fp16": "core", "atom_attn": "core", "pf_attn": "core", "opm_fused": "msa", "pwa_fused": "msa"}
PACKAGE = "opt_core.kernels.apb"                                           # the attention levers' provider
IMPORT_NAME = "opt_core"                                                   # whose __version__ the levers' record carries
MSA_IMPORT_NAME = "protenix_fpf_msa"                                       # the MSA-module package (opm_fused / pwa_fused), carried under third_party/
STRATEGIES: Dict[str, str] = {"dit_attn": "F5.flash_attn_dense", "dit_attn_fp16": "F4.autocast_policy", "atom_attn": "F5.flash_attn_dense",
                              "pf_attn": "LOCAL.protenix_v2.pf_attn", "opm_fused": "LOCAL.protenix_v2.opm_fused", "pwa_fused": "LOCAL.protenix_v2.pwa_fused"}
DTYPE_GATED: Tuple[str, ...] = ("pf_attn", "opm_fused", "pwa_fused")            # kernels that serve the bf16-autocast statement only: gated per call (dtype_gate), aside BY NAME otherwise
GATED_ATTRS: Tuple[str, ...] = ("forward", "standard_multihead_attention")          # the instance attributes the packages' installs bind (fpf_msa: forward; fpf_apb pf_attn: standard_multihead_attention)
REQUIRES_MSG = "dit_attn_fp16: requires dit_attn (PTX_DIT_ATTN=1) — the precision lever selects the operands of dit_attn's kernel"   # the package's own wording (INTEGRATION §10)
TESTED_TRITON: Tuple[str, ...] = ("3.7",)                                  # triton major.minor the cells were measured on; another one engages and is NAMED
_STATE: Dict[str, Any] = {"on": [], "patch": None, "installed": {}, "errors": {}, "named": {}, "precision": None, "models": 0, "gate": {}}   # gate: {lever: dtype_gate tally}


def from_env(lever: str, environ=None) -> bool:
    """Whether the lever's switch turns it on ("1"); "0" / unset / empty = off; any other value is refused by name (the package's grammar:
    ``PTX_DIT_ATTN='2': expected 0 or 1``)."""
    env = ENVS[lever]
    v = (environ if environ is not None else os.environ).get(env)
    if v is None or v == "" or v == "0":
        return False
    if v == "1":
        return True
    raise ValueError(f"{env}={v!r}: expected 0 or 1")


def check_requires(on: List[str]) -> None:
    """A precision lever on without its kernel lever: refused by name (RuntimeError, the package's wording) — never a silent no-op."""
    for prec, kernel in PRECISION.items():
        if prec in on and kernel not in on:
            raise RuntimeError(REQUIRES_MSG if prec == "dit_attn_fp16" else f"{prec}: requires {kernel}")


def triton_note() -> Optional[str]:
    """``untested triton <v>: engaged with the <tested> cells`` when the installed triton's major.minor is not a measured one; None otherwise
    (or when triton's version cannot be read here — the unit names a triton it cannot use by itself)."""
    from opt_core.gates import dist_version
    v = dist_version("triton")
    if not v:
        return None
    mm = ".".join(v.split(".")[:2])
    return None if mm in TESTED_TRITON else f"untested triton {v}: engaged with the {'/'.join(TESTED_TRITON)} cells"


def _package(lever: str = "dit_attn"):
    """The module that installs `lever`: ``protenix_opt.apb_core`` (the attention levers: the shared core's provider by tier word) or the
    carried ``protenix_fpf_msa`` package (imports torch and triton). A missing module raises by name — never a silent stock path."""
    imp, own = PACKAGES[LEVER_PKG[lever]]
    try:
        pkg = importlib.import_module(imp)
    except ImportError as e:
        where = ("the kit package itself" if LEVER_PKG[lever] == "core"
                 else f"opt/forward/flashpairformer/third_party/{imp}, reached through env.sh's PYTHONPATH")
        raise RuntimeError(f"apb_levers: {imp} ({own}) is not importable ({e!r}); it is carried at {where}") from e
    return pkg


def _version(lever: str) -> str:
    """The version word the lever's marker line carries: the shared core's for the provider-bound levers, the MSA package's own otherwise."""
    if LEVER_PKG[lever] == "core":
        from . import apb_core
        return apb_core.core_version()
    return str(getattr(sys.modules.get(MSA_IMPORT_NAME), "__version__", None) or getattr(_package(lever), "__version__", "?"))


def on_line(lever: str, st: dict, version: str, named: List[str]) -> str:
    """The marker grammar for an installed kernel lever: ``<MARK>on(<n> modules; cell=<cc> <k=v ...>; <provider|package> <version>[; NAMED <text>])``."""
    cell = st.get("cell") or {}
    kv = " ".join(f"{k}={v}" for k, v in sorted(cell.items()))
    return (f"{MARKS[lever]}on({int(st.get('installed_on') or 0)} modules; cell={st.get('cell_key')}" + (f" {kv}" if kv else "")
            + f"; {PACKAGES[LEVER_PKG[lever]][1]} {version}" + (f"; NAMED {'; '.join(named)}" if named else "") + ")")


def gate_tally(lever: str) -> dict:
    """The dtype gate's tally of `lever` in this process (``dtype_gate.new_tally`` form; created on first use, kept across runners)."""
    from .dtype_gate import new_tally
    g = _STATE.setdefault("gate", {})
    if lever not in g:
        g[lever] = new_tally()
    return g[lever]


def gate_installed(lever: str, model) -> int:
    """Re-wrap, from the outside, every callable the package's install of `lever` bound on a module of `model` (an instance attribute of
    GATED_ATTRS carrying the package's ownership mark for this lever) with the per-call dtype gate (``dtype_gate.gate_method``); returns the
    number of sites gated. Idempotent (a site already gated is left as it is); a lever outside DTYPE_GATED gates nothing."""
    if lever not in DTYPE_GATED or model is None or not callable(getattr(model, "modules", None)):
        return 0
    import torch
    from .dtype_gate import gate_method
    tally = gate_tally(lever)
    n = 0
    for mod in model.modules():
        for attr in GATED_ATTRS:
            fn = mod.__dict__.get(attr)
            if fn is None or getattr(fn, "_ptx_dtype_gate", None) or lever not in (getattr(fn, "_fpf_msa", None), getattr(fn, "_fpf_apb", None)):
                continue
            setattr(mod, attr, gate_method(mod, attr, fn, tally, torch, lever)); n += 1
    return n


def gate_census() -> Dict[str, Tuple[Optional[bool], str]]:
    """``{lever: (ok, detail)}`` for the gated levers switched on in this process once a runner was built (``dtype_gate.census`` on each tally):
    the verdict's served-call reading (stack.served_census)."""
    from .dtype_gate import census
    if not _STATE.get("on") or not _STATE.get("gate"):
        return {}
    return {lever: census(_STATE["gate"].get(lever)) for lever in DTYPE_GATED if lever in _STATE["on"] and lever in _STATE["gate"]}


def _installer(lever: str):
    """The runner-seam installer of a kernel lever: ``apb_core.install_<lever>(runner.model, word=<tier word>)`` (dit_attn: ``fp16=<dit_attn_fp16
    on>``, the one install point of both) or ``fpf_msa.install_<lever>(runner.model)``; names the result (or the failure) on the kit's stream.""" 

    def install_on(runner) -> None:
        pkg = _package(lever)
        fp16 = lever == "dit_attn" and "dit_attn_fp16" in _STATE["on"]
        try:
            if LEVER_PKG[lever] == "core":            # every card: the shared core's provider by the mode's tier word (apb_core); no kit cell / tile table
                word = pkg.tier_word()
                fn = pkg.INSTALLERS[lever]
                st = fn(runner.model, word=word, fp16=fp16) if lever == "dit_attn" else fn(runner.model, word=word)
            else:
                fn = getattr(pkg, INSTALL_FNS[lever], None) or getattr(importlib.import_module(PACKAGES[LEVER_PKG[lever]][0] + ".install"), INSTALL_FNS[lever])
                st = fn(runner.model)
        except Exception as e:                       # a lever that cannot engage here STEPS ASIDE BY NAME: the modules keep the stock statement, the stream says why, the run goes on; reconcile names it (levers_fallback + reason) from this record
            _STATE["errors"][lever] = repr(e)
            _STATE["installed"][lever] = 0
            _STATE["named"][lever] = f"aside at install: {e!r}"
            sys.stderr.write(f"{PREFIX} {MARKS[lever]}aside(install: {e!r}; the modules keep the stock statement)\n")
            if lever == "dit_attn":
                _STATE["precision"] = "fp32"
                sys.stderr.write(f"{PREFIX} {MARKS['dit_attn_fp16']}off\n")
            sys.stderr.flush()
            return
        if st.get("aside"):                          # a state with no binding on this card steps aside BY NAME: the modules keep the stock statement, the line says why (`skipped(`: stack.BAD, so the lever reads as a named fallback, never as on)
            _STATE["named"][lever] = st["aside"]
            sys.stderr.write(f"{PREFIX} {MARKS[lever]}skipped({st['aside']})\n")
            if lever == "dit_attn":
                _STATE["precision"] = st.get("precision") or ("fp16" if fp16 else "fp32")
                sys.stderr.write(f"{PREFIX} {MARKS['dit_attn_fp16']}{'on' if fp16 else 'off'}\n")
            sys.stderr.flush()
            return
        gate_installed(lever, getattr(runner, "model", None))          # DTYPE_GATED: the per-call dtype gate around what the install bound (aside BY NAME outside bf16 autocast, counted)
        left = int(st.get("left_alone") or 0)                         # modules another lever already owns (an instance-level forward): left alone and NAMED (the MSA package's rule)
        named = [n for n in (st.get("named"), (f"{left} module(s) owned by another lever left alone" if left else None), triton_note()) if n]
        _STATE["installed"][lever] = int(st.get("installed_on") or 0)
        _STATE["named"][lever] = "; ".join(named) if named else None
        sys.stderr.write(f"{PREFIX} {on_line(lever, st, _version(lever), named)}\n")
        if lever == "dit_attn":                      # the precision lever's state rides the kernel lever's install: its own marker line
            _STATE["precision"] = st.get("precision") or ("fp16" if fp16 else "fp32")
            sys.stderr.write(f"{PREFIX} {MARKS['dit_attn_fp16']}{'on' if fp16 else 'off'}\n")
        sys.stderr.flush()

    install_on.__name__ = f"install_{lever}"
    return install_on


def install(levers: List[str]) -> List[str]:
    """Arm the named levers (a precision lever without its kernel lever raises by name first): each kernel lever's installer registered on the
    runner seam (after whatever registered before — sampler_fuse) and the seam patched now or at the runner module's import. Returns the
    applied markers in lever order: ``<FAMILY>armed|patched`` for the kernel levers, ``DIT_ATTN_FP16:requested`` for the precision lever
    (its install-time line reads ``DIT_ATTN_FP16:on``)."""
    from . import runner_seam
    check_requires(list(levers))
    out = []
    for lever in LEVERS:
        if lever not in levers:
            continue
        if lever not in _STATE["on"]:
            _STATE["on"].append(lever)
        if lever in KERNEL_LEVERS:
            runner_seam.add(lever, _installer(lever))
            _STATE["patch"] = runner_seam.arm()
            out.append(f"{MARKS[lever]}{'patched' if runner_seam.patched() else 'armed'}")
        else:
            out.append(f"{MARKS[lever]}requested")
    return out


def package_report() -> Dict[str, dict]:
    """The loaded packages' report()s merged ({lever: {...}}); {} for a package not loaded in this process (nothing is imported to ask)."""
    out: Dict[str, dict] = {}
    for imp, own in PACKAGES.values():
        pkg = sys.modules.get(imp)
        rep = getattr(pkg, "report", None) if pkg is not None else None
        if rep is None and pkg is not None:
            rep = getattr(sys.modules.get(imp + ".install"), "report", None)
        if rep is None:
            continue
        if imp == "protenix_opt.apb_core":
            rep = getattr(pkg, "lever_report", None)                           # apb_core.report() is dit_attn_exact's hook record; the runner-seam levers' is lever_report()
        try:
            out.update(dict(rep()))
        except Exception as e:  # noqa: BLE001 — a report that raises is recorded, not fatal at exit
            out[own + "_report_error"] = {"error": repr(e)}
    return out


def state() -> dict:
    """The levers' end-of-run record: per kernel lever whether it was switched on, modules installed on, Python-level calls (under the sampler
    graph: the eager warm-up + capture calls; replays run the captured kernels), the cell, NAMED conditions, errors; for the precision lever
    whether it was requested and the precision state the kernel lever's install reports (``fp16`` with it, ``fp32`` = tf32x3 without)."""
    from . import runner_seam
    rep = package_report()
    out: Dict[str, Any] = {"patched": runner_seam.patched(), "models": runner_seam.state()["runners"], "package": PACKAGE,
                           "version": getattr(sys.modules.get(IMPORT_NAME), "__version__", None),
                           "msa_version": getattr(sys.modules.get(MSA_IMPORT_NAME), "__version__", None)}
    for lever in KERNEL_LEVERS:
        r = rep.get(lever) or {}
        out[lever] = {"on": lever in _STATE["on"], "installed_on": int(r.get("installed_on") or _STATE["installed"].get(lever) or 0),
                      "calls": int(r.get("calls") or 0), "cell_key": r.get("cell_key"), "cell": r.get("cell"),
                      "named": _STATE["named"].get(lever) or r.get("named"), "error": _STATE["errors"].get(lever) or r.get("error"),
                      "extra": {k: r.get(k) for k in ("chunks", "zcache_hits", "left_alone") if r.get(k) is not None},
                      "served": dict(r.get("served") or {}), "aside_kinds": dict(r.get("aside_kinds") or {}), "producer": dict(r.get("producer") or {}),
                      "gate": dict((_STATE.get("gate") or {}).get(lever) or {}) or None}
    out["dit_attn"]["precision"] = (rep.get("dit_attn") or {}).get("precision") or _STATE.get("precision")
    pf = rep.get("dit_attn_fp16") or {}
    out["dit_attn_fp16"] = {"on": "dit_attn_fp16" in _STATE["on"], "engaged": bool(pf.get("on")) if pf else (_STATE.get("precision") == "fp16"),
                            "engaged_calls": int(pf.get("engaged_calls") or 0), "kernel": "dit_attn", "error": pf.get("error")}
    return out


def _rows_token(d: dict) -> str:
    """``arm:n+arm:n`` (sorted by arm) for a {arm: count} tally; ``none`` when empty (a whitespace-free LEVER value)."""
    return "+".join(f"{_token(k)}:{int(v)}" for k, v in sorted(d.items())) if d else "none"


def _count(v) -> Any:
    """An int when the value is one (or parses as one), else a whitespace-free token — a LEVER pair never raises on a package's value."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return _token(v)


def evidence(lever: str) -> list:
    """The (key, value) pairs the lever's LEVER line carries after impl/origin: kernel levers — modules installed on, calls, cell, operand class,
    precision state (dit_attn), NAMED; the precision lever — the kernel lever it modifies, whether the install engaged fp16, the operands."""
    st = state()
    if lever in PRECISION:
        r = st[lever]; k = st[PRECISION[lever]]
        return [("models", st["models"]), ("kernel", PRECISION[lever]), ("engaged", int(bool(r["engaged"]))), ("opd", "fp16"),
                ("kernel_modules", k["installed_on"]), ("kernel_calls", k["calls"]), ("fp16_calls", int(r.get("engaged_calls") or 0))]
    r = st.get(lever) or {}; cell = r.get("cell") or {}
    pairs = [("models", st.get("models", 0)), ("modules", int(r.get("installed_on") or 0)), ("calls", int(r.get("calls") or 0)), ("cell", r.get("cell_key") or "none")]
    if lever == "dit_attn":
        pairs.append(("precision", r.get("precision") or "none"))
    pairs += [(k, (int(v) if isinstance(v, bool) else _token(str(v)))) for k, v in sorted(cell.items())]
    for k, v in sorted((r.get("extra") or {}).items()):        # the packages' extra counters: ints (chunks, zcache_hits) or a list of module names (left_alone) -> its length
        pairs.append((k, len(v) if isinstance(v, (list, tuple, set, dict)) else _count(v)))
    from .dtype_gate import aside_token
    aside = aside_token(r.get("gate"))
    if aside:                                                   # DTYPE_GATED: calls that stepped aside BY NAME outside bf16 autocast (`aside=dtype_fp32:<n>`); absent on a run where none did
        pairs.append(("aside", aside))
    elif r.get("aside_kinds"):                                  # the provider-bound levers: calls no admitted row served took the statement BY NAME (`aside=<kind>:<n>+...`)
        pairs.append(("aside", _rows_token(r["aside_kinds"])))
    pairs.append(("named", _token(r["named"]) if r.get("named") else "none"))
    if LEVER_PKG.get(lever) == "core":                          # the provider-bound levers: calls served through the face and the arms the table chose (trailing pairs)
        pairs.append(("served", int(sum(int(v) for v in (r.get("served") or {}).values()))))
        pairs.append(("rows", _rows_token(r.get("served") or {})))
        if lever == "pf_attn":
            pairs.append(("bias_rows", _rows_token(r.get("producer") or {})))
    return pairs
