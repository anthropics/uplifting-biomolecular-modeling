"""The diffusion sampler's fused-stack levers — ``cond_dedupe``, ``dit_fused``, ``dit_lowp`` (third_party/protenix_fpf_ditfast) and the
exact composition's ``atom_attn_exact`` (``protenix_opt.apb_atom_exact`` on the shared core's provider) — plus ``atom_fused`` (protenix_fpf_ditfast), on the
runner seam (``runner_seam``: ONE patch of ``InferenceRunner.__init__``; these installers run AFTER sampler_fuse and the sampler attention
levers of ``apb_levers``, the order the packages state: graphed sampler + DiT hoist at init_model, DIT_FUSE, dit_attn / atom_attn, then
these).

Switches (0|1 unless stated; any other value refused by name; a switch set under a mode whose row does not list the lever is refused by name
in ``stack._sampler_levers_on``): ``PTX_COND_DEDUPE`` (cond_dedupe), ``PTX_DIT_FAST`` (dit_fused; requires dit_attn — the package refuses by
name without it), ``PTX_DIT_LOWP`` = ``off`` | ``fp16`` | ``bf16`` (dit_lowp, the PRECISION lever riding dit_fused: the token-block a-path
GEMM operands' dtype; ``fp16`` is the measured cell word, ``bf16`` the documented alternative; a word other than off without dit_fused is refused
by name), ``PTX_ATOM_FAST`` (atom_fused; requires atom_attn), ``PTX_ATOM_ATTN_EXACT`` (atom_attn_exact; the exact composition's rows only — the
package refuses by name when atom_attn or atom_fused own the site).

Markers. At activation (the seam armed, no runner yet): ``<MARK>armed|patched`` (``CONDDEDUPE:``, ``DITFAST:``, ``ATOMFAST:``,
``ATOMATTNEXACT:``) and ``DITLOWP:requested``. At install the packages print their own lines on stderr (``[protenix_fpf_ditfast]
CONDDEDUPE:on(guard=stride0)``, ``[protenix_fpf_ditfast] DITFAST:on(blocks=24 act=… lowp=fp16 …)``, ``[protenix_fpf_ditfast] ATOMFAST:on(enc=3
dec=3 …)``, ``[protenix_fpf_atom_attn_exact] ATOMATTNEXACT:on(routes=… loadcheck=4/4 …)``) and the kit echoes ``[protenix-opt]
<MARK>installed(<n> sites; <package> <version>)`` plus ``[protenix-opt] DITLOWP:on(word=<w>)|off``; a failed install prints ``[protenix-opt]
<MARK>unavailable(<repr>)`` — ``CONDDEDUPE:unavailable(``, ``DITFAST:unavailable(``, ``ATOMFAST:unavailable(``, ``ATOMATTNEXACT:unavailable(`` —
and the error propagates (the package's ``LeverRefused`` by name; no stock fallback exists inside the packages). At exit the packages print
``[protenix_fpf_ditfast] EXIT <lever> {…}`` / ``[protenix_fpf_atom_attn_exact] EXIT atom_attn_exact {…}`` (their census; also one JSON line in
``$PTX_LEVER_REPORT``); the kit's LEVER lines carry the same counters (``evidence``).

Dtype / envelope (``dtype_gate``): ``atom_attn_exact``'s kernel serves the chunk batch counts whose cuBLAS numerics its
install determined; a call outside them raises the package's ``kernel.Refused`` BY NAME. The kit's binding catches it PER CALL and answers with
the stock ``_local_attention`` the install displaced, counted under the refusal's word (``aside=cublas_route_full_chunk:<n>`` on the LEVER line;
the package's ``kernel`` count is kept to the calls its kernel served) — under stock's ``--dtype fp32`` the input embedder's atom encoder call
(fp32 there only then) enters the kernel's dtype envelope with an un-chunked batch count. The other levers here serve the diffusion sampler, fp32
under every ``--dtype`` (stock's ``skip_amp.sample_diffusion``): no dtype gate.
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PREFIX = "[protenix-opt]"
LEVERS: Tuple[str, ...] = ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact")      # registry names, in install order
KERNEL_LEVERS: Tuple[str, ...] = ("cond_dedupe", "dit_fused", "atom_fused", "atom_attn_exact")           # the levers with an installer of their own
PRECISION: Dict[str, str] = {"dit_lowp": "dit_fused"}                                                     # precision lever -> the kernel lever whose install applies it
ENVS: Dict[str, str] = {"cond_dedupe": "PTX_COND_DEDUPE", "dit_fused": "PTX_DIT_FAST", "dit_lowp": "PTX_DIT_LOWP", "atom_fused": "PTX_ATOM_FAST",
                        "atom_attn_exact": "PTX_ATOM_ATTN_EXACT"}
LOWP_WORDS: Tuple[str, ...] = ("fp16", "bf16")                                                            # dit_lowp's on-words (fp16 = the measured cell; bf16 = the documented alternative)
LOWP_OFF = "off"
MARKS: Dict[str, str] = {"cond_dedupe": "CONDDEDUPE:", "dit_fused": "DITFAST:", "dit_lowp": "DITLOWP:", "atom_fused": "ATOMFAST:", "atom_attn_exact": "ATOMATTNEXACT:"}
PACKAGES: Dict[str, Tuple[str, str]] = {"ditfast": ("protenix_fpf_ditfast", "fpf_ditfast"), "atomx": ("protenix_opt.apb_atom_exact", "kernels.apb.atom_exact")}   # key -> (import name, the name on the kit's lines)
LEVER_PKG: Dict[str, str] = {"cond_dedupe": "ditfast", "dit_fused": "ditfast", "dit_lowp": "ditfast", "atom_fused": "ditfast", "atom_attn_exact": "atomx"}
INSTALL_FNS: Dict[str, str] = {"cond_dedupe": "install_cond_dedupe", "dit_fused": "install_dit_fast", "atom_fused": "install_atom_fast", "atom_attn_exact": "install"}
STRATEGIES: Dict[str, str] = {"cond_dedupe": "LOCAL.protenix_v2.cond_dedupe", "dit_fused": "LOCAL.protenix_v2.dit_fused", "dit_lowp": "F4.autocast_policy",
                              "atom_fused": "LOCAL.protenix_v2.atom_fused", "atom_attn_exact": "LOCAL.protenix_v2.atom_attn_exact"}

_STATE: Dict[str, Any] = {"on": [], "patch": None, "installed": {}, "errors": {}, "reports": {}, "lowp": None, "models": 0, "gate": {}}   # gate: {lever: dtype_gate tally}


def from_env(lever: str, environ=None) -> bool:
    """Whether the lever's switch turns it on. 0|1 switches: "1" on, "0" / unset / empty off, anything else refused by name
    (``PTX_DIT_FAST='2': expected 0 or 1``). dit_lowp's word: ``fp16`` | ``bf16`` on, ``off`` / unset / empty off, anything else refused by name."""
    env = ENVS[lever]
    v = (environ if environ is not None else os.environ).get(env)
    if lever == "dit_lowp":
        if v is None or v == "" or v == LOWP_OFF:
            return False
        if v in LOWP_WORDS:
            return True
        raise ValueError(f"{env}={v!r}: expected {LOWP_OFF}, {' or '.join(LOWP_WORDS)}")
    if v is None or v == "" or v == "0":
        return False
    if v == "1":
        return True
    raise ValueError(f"{env}={v!r}: expected 0 or 1")


def lowp_word(environ=None) -> str:
    v = (environ if environ is not None else os.environ).get(ENVS["dit_lowp"])
    return v if v in LOWP_WORDS else LOWP_OFF


def check_requires(on: List[str]) -> None:
    """A precision lever on without its kernel lever: refused by name — never a silent no-op (the package reads PTX_DIT_LOWP only inside
    install_dit_fast, so a word set with PTX_DIT_FAST=0 would otherwise vanish)."""
    for prec, kernel in PRECISION.items():
        if prec in on and kernel not in on:
            raise RuntimeError(f"{prec}: requires {kernel} ({ENVS[prec]}={lowp_word()} with {ENVS[kernel]} != 1)")


def _package(lever: str):
    return importlib.import_module(PACKAGES[LEVER_PKG[lever]][0])


def installed_sites(lever: str, st: dict) -> int:
    """How many sites an install report says the lever holds: cond_dedupe 1 (DiffusionConditioning.forward), dit_fused its token blocks,
    atom_fused its encoder + decoder blocks, atom_attn_exact 1 (primitives._local_attention); 0 when the report says not installed."""
    if not st or not st.get("installed"):
        return 0
    if lever == "dit_fused":
        return int(st.get("blocks") or 0)
    if lever == "atom_fused":
        stacks = st.get("stacks") or {}
        return int(sum(int((v or {}).get("blocks") or 0) for v in stacks.values())) if isinstance(stacks, dict) else 0
    return 1


GATED: Tuple[str, ...] = ("atom_attn_exact",)                              # the lever whose per-call Refused the binding answers with the stock statement (dtype_gate.gate_refusable)
ATOMX_SITE = ("protenix.model.modules.primitives", "_local_attention")     # the module global the atom_attn_exact package's install binds (kernels.apb row atom_exact via protenix_opt.apb_atom_exact)


def gate_tally(lever: str) -> dict:
    """The gate tally of `lever` in this process (``dtype_gate.new_tally`` form; created on first use)."""
    from .dtype_gate import new_tally
    g = _STATE.setdefault("gate", {})
    if lever not in g:
        g[lever] = new_tally()
    return g[lever]


def _atomx_site():
    """(module, current callable) of atom_attn_exact's site, or (None, None) when protenix's primitives are not importable here."""
    try:
        mod = importlib.import_module(ATOMX_SITE[0])
    except ImportError:
        return None, None
    return mod, getattr(mod, ATOMX_SITE[1], None)


def gate_atom_attn_exact(prior) -> bool:
    """Re-wrap, from the outside, the fused ``_local_attention`` the package's install put on protenix's primitives: a call the kernel refuses BY
    NAME (``kernel.Refused``: a chunk batch count whose cuBLAS numerics the install did not determine) is answered by ``prior`` — the stock statement
    the install displaced —, counted under the refusal's word; the package's own ``calls['kernel']`` (incremented before its route check) is taken
    back by one so its census keeps counting what the kernel served. Idempotent; False when there is nothing to gate (site absent, not the
    package's callable, no stock callable to hand the call to)."""
    mod, fused = _atomx_site()
    if mod is None or fused is None or not getattr(fused, "_atom_attn_exact", False):
        return False
    if getattr(fused, "_ptx_dtype_gate", None):
        return True
    if prior is None or prior is fused or getattr(prior, "_atom_attn_exact", False):
        return False
    _pkg = _package("atom_attn_exact")
    K = getattr(_pkg, "kernel", None) or importlib.import_module(PACKAGES["atomx"][0] + ".kernel")   # the provider's carried kernel module (kernels.apb row atom_exact): its Refused word + call tallies
    from .dtype_gate import gate_refusable

    def on_aside(word, _K=K):
        calls = (getattr(_K, "STATE", None) or {}).get("calls")
        if isinstance(calls, dict) and int(calls.get("kernel") or 0) > 0:
            calls["kernel"] -= 1

    setattr(mod, ATOMX_SITE[1], gate_refusable(fused, prior, (K.Refused,), gate_tally("atom_attn_exact"), on_aside=on_aside))
    return True


def gate_census() -> Dict[str, Tuple[Optional[bool], str]]:
    """``{lever: (ok, detail)}`` for the gated levers switched on in this process (``dtype_gate.census``): the verdict's served-call reading."""
    from .dtype_gate import census
    if not _STATE.get("on") or not _STATE.get("gate"):
        return {}
    return {lever: census(_STATE["gate"].get(lever)) for lever in GATED if lever in _STATE["on"] and lever in _STATE["gate"]}


def _installer(lever: str):
    """The runner-seam installer of a kernel lever: ``<package>.<install fn>(runner.model)``; names the result (or the failure) on the kit's
    stream; an error propagates (the run stops by name)."""

    def install_on(runner) -> None:
        pkg = _package(lever)
        fn = getattr(pkg, INSTALL_FNS[lever])
        prior = _atomx_site()[1] if lever in GATED else None       # the stock statement at the site before the package binds its kernel (the aside's target)
        try:
            st = fn(runner.model)
        except Exception as e:                       # named on the kit's stream, then raised: no stock fallback
            _STATE["errors"][lever] = repr(e)
            sys.stderr.write(f"{PREFIX} {MARKS[lever]}unavailable({e!r})\n"); sys.stderr.flush()
            raise
        st = dict(st or {})
        if lever in GATED:
            st["gated"] = gate_atom_attn_exact(prior)              # the per-call Refused -> stock statement gate around the bound kernel (dtype_gate)
        _STATE["reports"][lever] = st
        _STATE["installed"][lever] = installed_sites(lever, st)
        sys.stderr.write(f"{PREFIX} {MARKS[lever]}installed({_STATE['installed'][lever]} sites; {PACKAGES[LEVER_PKG[lever]][1]} {getattr(pkg, '__version__', '?')})\n")
        if lever == "dit_fused":                     # the precision lever rides this install: its own kit line
            _STATE["lowp"] = st.get("lowp") or lowp_word()
            _bg = sys.modules.get(__package__ + ".big")
            if _bg is not None and _bg._STATE.get("apb_policy") is not None:      # big (apb_bias_chunk applied): the stack's 24 block biases produced one block at a time (big.dit_bias_per_block)
                _n = _bg.dit_bias_per_block(runner.model)
                sys.stderr.write(f"{PREFIX} MEM:apb_bias_chunk(dit_stacks_per_block={_n}; bound after dit_fused's install)\n")
            on = "dit_lowp" in _STATE["on"] and _STATE["lowp"] in LOWP_WORDS
            sys.stderr.write(f"{PREFIX} {MARKS['dit_lowp']}{'on(word=' + str(_STATE['lowp']) + ')' if on else 'off'}\n")
        sys.stderr.flush()

    install_on.__name__ = f"install_{lever}"
    return install_on


def install(levers: List[str]) -> List[str]:
    """Arm the named levers on the runner seam (after whatever registered before: sampler_fuse, apb_levers). Returns the applied markers
    in lever order: ``<MARK>armed|patched`` for the kernel levers, ``DITLOWP:requested`` for the precision lever."""
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
    """The loaded packages' reports ({lever: {...}}): protenix_fpf_ditfast.report() = its EXIT census per lever; the atom_attn_exact
    package's report(); {} for a package not loaded in this process."""
    out: Dict[str, dict] = {}
    d = sys.modules.get(PACKAGES["ditfast"][0])
    if d is not None:
        try:
            rep = d.report() or {}
        except Exception as e:   # noqa: BLE001 — a broken report is named, never fatal at exit
            rep = {"error": repr(e)}
        for lever in ("cond_dedupe", "dit_fused", "atom_fused"):
            if isinstance(rep.get(lever), dict):
                out[lever] = rep[lever]
    x = sys.modules.get(PACKAGES["atomx"][0])
    if x is not None:
        try:
            out["atom_attn_exact"] = dict(x.report() or {})
        except Exception as e:   # noqa: BLE001
            out["atom_attn_exact"] = {"error": repr(e)}
    return out


def state() -> dict:
    """The levers' end-of-run record: per kernel lever whether it was switched on, the sites installed on, its census (the package's EXIT
    dict), errors; for dit_lowp whether it was requested and the word dit_fused's install engaged."""
    from . import runner_seam
    rep = package_report()
    out: Dict[str, Any] = {"patched": runner_seam.patched(), "models": runner_seam.state()["runners"],
                           "versions": {k: getattr(sys.modules.get(v[0]), "__version__", None) for k, v in PACKAGES.items()}}
    for lever in KERNEL_LEVERS:
        out[lever] = {"on": lever in _STATE["on"], "installed_on": int(_STATE["installed"].get(lever) or 0),
                      "census": rep.get(lever) or {}, "install": _STATE["reports"].get(lever) or {}, "error": _STATE["errors"].get(lever),
                      "gate": dict((_STATE.get("gate") or {}).get(lever) or {}) or None}
    word = _STATE.get("lowp") or ((rep.get("dit_fused") or {}).get("lowp"))
    out["dit_lowp"] = {"on": "dit_lowp" in _STATE["on"], "engaged": word in LOWP_WORDS, "word": word or LOWP_OFF, "kernel": "dit_fused"}
    return out


def _token(v) -> str:
    s = str(v)
    for a, b in ((" ", ""), ("'", ""), ('"', ""), ("{", ""), ("}", ""), ("[", ""), ("]", "")):
        s = s.replace(a, b)
    return s or "none"


def evidence(lever: str) -> List[Tuple[str, Any]]:
    """LEVER-line pairs (whitespace-free key=value): cond_dedupe dedupe_calls / stock_path_calls / rows_in / rows_computed; dit_fused blocks /
    stack_calls / blocks_x_calls / lowp / bias_slots; dit_lowp word / engaged; atom_fused blocks / stack_calls / blocks_x_calls / cond_slots;
    atom_attn_exact kernel_calls / original_calls / original_reasons / loadcheck / routes. ``sites=0`` + empty counters when no runner was
    built in this process (a dry run, a launcher)."""
    st = state()
    if lever == "dit_lowp":
        r = st["dit_lowp"]
        return [("word", r["word"]), ("engaged", int(bool(r["engaged"]))), ("kernel", "dit_fused")]
    r = st.get(lever) or {}
    cen, ins = (r.get("census") or {}), (r.get("install") or {})
    pairs: List[Tuple[str, Any]] = [("models", st.get("models", 0)), ("sites", int(r.get("installed_on") or 0))]
    if lever == "cond_dedupe":
        pairs += [(k, int(cen.get(k) or 0)) for k in ("dedupe_calls", "stock_path_calls", "rows_in", "rows_computed")]
    elif lever == "dit_fused":
        pairs += [("stack_calls", int(cen.get("stack_calls") or 0)), ("blocks_x_calls", int(cen.get("blocks_x_calls") or 0)),
                  ("lowp", cen.get("lowp") or ins.get("lowp") or "none"), ("bias_slots", cen.get("bias_slots") or ins.get("bias_slots") or "none"),
                  ("act", _token(cen.get("act") or ins.get("act") or "none"))]
    elif lever == "atom_fused":
        sc = cen.get("stack_calls") or {}
        pairs += [("stack_calls", _token(",".join(f"{k}:{v}" for k, v in sorted(sc.items())) if isinstance(sc, dict) else sc)),
                  ("blocks_x_calls", int(cen.get("blocks_x_calls") or 0)), ("cond_slots", cen.get("cond_slots") or ins.get("cond_slots") or "none")]
    elif lever == "atom_attn_exact":
        calls = cen.get("calls") or {}
        why = cen.get("original_path_reasons") or {}
        pairs += [("kernel_calls", int(calls.get("kernel") or 0)), ("original_calls", int(calls.get("original") or 0)),
                  ("original_reasons", _token(",".join(f"{k}:{v}" for k, v in sorted(why.items()))) if why else "none"),
                  ("loadcheck", _token(len(ins.get("loadcheck") or [])) if ins else "none"), ("routes", _token(cen.get("routes") or ins.get("routes") or "none"))]
        from .dtype_gate import aside_token
        aside = aside_token(r.get("gate"))
        if aside:                                               # calls the kernel refused BY NAME, answered with the stock statement (`aside=cublas_route_full_chunk:<n>`); absent when none did
            pairs.append(("aside", aside))
    if r.get("error"):
        pairs.append(("error", _token(r["error"])[:120]))
    return pairs
