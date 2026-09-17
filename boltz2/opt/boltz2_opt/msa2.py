"""boltz2_opt.msa2 — the MSA track's engine adapter (BOLTZ2-FASTER, track MSA): second-generation MSA-module levers carried in ``opt/forward/msa2/``,
superseding the kit-carried fpf_msa kernels BY NAME where they overlap (registry words proposed in the LEVER_SPECs; the row owner assigns the
row word — one row env word per lever, apply() reports state=on|off|skipped with a reason by name, ``off`` = the exact stock statement).

Row word (proposed): ``BOLTZ_MSA2=<unit>[,<unit>...]`` — units:
    hoist      lever ``msa_hoist`` (tier 1, EXACT: bitwise = stock): msa2/exact_hoist.py — PairWeightedAveraging / OuterProductMean / the dim-64
               Transitions run their stock statement sequence with the fp32->bf16 activation cast done once per call instead of once per head / per
               hidden chunk (autocast re-casts activations at every matmul), OPM's num_mask built once per MSA-mask tensor instead of once per call, and
               OPM's fp32 round trip of the bf16 chunk result elided (autocast casts it straight back). Served in the bit-safe regime only (eval + CUDA
               autocast bf16), else the stock statements BY NAME (``training`` / ``autocast_off`` / ``autocast_dtype``), counted.
    trans2     lever ``msa_trans2`` (tier 2, FAST numerics class): msa2/trans2.py — the dim-64/hidden-256 Transition (MSA transition + template
               pairformer transition) as ONE fused Triton kernel per call (LN -> fc1|fc2 -> SiLU*mul -> fc3 on-chip, hidden processed in register
               chunks, fp32 accumulation of the fc3 partials; stock: ~40 HBM passes per call). Served for bf16 CUDA inputs in eval under autocast-bf16,
               else the previous forward BY NAME, counted. Not bitwise (rounding placement) => never in the exact row; its stock-rounding 'mimic' mode is
               a candidate, gated on a per-card bit-equality check (not yet passed).
    trans2x    lever ``msa_trans2_exact`` (tier 1, EXACT: bitwise = stock): the same kernel in stock-rounding mode (msa2/trans2.py MODE=1): the stock
               nn.LayerNorm statement, then ONE kernel = autocast's bf16 cast + fc1/fc2 slice GEMMs (tl.dot, bit-identical to cuBLAS's bf16 GEMM for
               K=64/32/256 at every M from 1 to 33.5M on sm90, checked exhaustively) + ATen's silu (x/(1+expf(-x)), div.rn,
               libdevice expf: exhaustively bit-identical over all bf16) + bf16 product + fc3 slice GEMMs + the bf16 partial-sum chain, in the caller's
               hidden chunking (32) or unchunked (K=256). Checked: torch.equal vs the stock module at 7 shapes + sha256 vs --mode off.
    pwa2x      lever ``msa_pwa_exact`` (tier 1, EXACT: bitwise = stock): the shared core's opt_core.ops.msa_pwa2 — PairWeightedAveraging as two Triton kernels (v|g projections +
               sigmoid in one pass; the K=N weighted average x gate x proj_o slice x the bf16 head chain in one kernel) around the stock LN / b / softmax
               statements. The K=N contraction reproduces whichever cuBLAS kernel family serves the class (nvjet: ascending k16 groups; cutlass_80 s16816
               align2: residue-first; cutlass_75 s1688 align1: k8 MMAs) — SELECTED PER (N, S, path) CLASS AT ITS FIRST EAGER ENCOUNTER: the stock statements
               run once, the candidates (pwa2.candidates(N)) are compared torch.equal, the match is locked for the process, no match = the class is served
               by the stock forward by name (bitcmp_failed:<class>). Pin-scoped like trans2x.
    probe      DEV INSTRUMENT ONLY (never in an exact/fast/big row): per-op CUDA-event attribution of the trunk (boltz2_opt.msa2_probe); numerics unchanged.

Contract (worker_launch.ATTACH "msa2", trigger boltz.model.models.boltz2, attached AFTER "transition"/"msa" so a unit that supersedes an earlier
adapter's class patch wraps it and hands it the calls it does not take): ``apply()`` -> list of applied unit names; ``report()`` -> dict under
``msa2_report`` in the worker log; one ``[boltz2-opt] LEVER name=<lever> ...`` line per lever on stderr at exit (dev evidence; report.py owns the
printed line at landing).
"""
from __future__ import annotations

import atexit
import importlib
import importlib.util
import os
import sys
from typing import Any, Dict, List

SWITCH = "BOLTZ_MSA2"
UNITS = ("hoist", "trans2", "trans2x", "pwa2x", "probe")
LEVERS = ("msa_hoist", "msa_trans2", "msa_trans2_exact", "msa_pwa_exact")
UNIT_LEVER = {"hoist": "msa_hoist", "trans2": "msa_trans2", "trans2x": "msa_trans2_exact", "pwa2x": "msa_pwa_exact"}
BUNDLE = "forward"                 # opt/-relative directory carrying the msa2 package
PACKAGE = "msa2"
_STATE: Dict[str, Any] = {"applied": [], "state": {}, "requested": [], "patched": [], "impl": {}}


def requested(env=None) -> List[str]:
    v = (env if env is not None else os.environ).get(SWITCH, "") or ""
    return [u.strip().lower() for u in v.split(",") if u.strip()]


def problems(env) -> List[str]:
    """Pre-launch refusal words (stack.attachment_problems): unknown unit names."""
    return [f"{SWITCH}={u}: not a unit of boltz2_opt.msa2 ({','.join(UNITS)})" for u in requested(env) if u not in UNITS]


def _load_package():
    """Import ``msa2`` as a package from opt/forward/msa2 (its own directory only; nothing else of opt/forward enters sys.path). Idempotent."""
    from . import stack
    if PACKAGE not in sys.modules:
        pkg_dir = stack.kit_path(f"{BUNDLE}/{PACKAGE}")
        spec = importlib.util.spec_from_file_location(PACKAGE, os.path.join(pkg_dir, "__init__.py"), submodule_search_locations=[pkg_dir])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"{PACKAGE}: no importable package at {pkg_dir}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = mod
        spec.loader.exec_module(mod)
    return sys.modules[PACKAGE]


def _apply_hoist() -> Dict[str, Any]:
    _load_package()
    H = importlib.import_module(f"{PACKAGE}.exact_hoist")
    from boltz.model.layers import pair_averaging as PWA, outer_product_mean as OPM, transition as TRN
    prev = {"pwa": PWA.PairWeightedAveraging.forward, "opm": OPM.OuterProductMean.forward, "transition": TRN.Transition.forward}
    PWA.PairWeightedAveraging.forward = H.pwa_forward(prev["pwa"])
    OPM.OuterProductMean.forward = H.opm_forward(prev["opm"])
    # the dim-64 Transitions (MSA transition 64->256->64, the template pairformer's transition): the core's cell table does not serve (64, 256) today
    # (transition adapter census `no-cell:transition:64x256` / `chunked`), so these calls ran the stock statements before this lever; every other
    # Transition call goes to whatever forward was installed before this one (the transition adapter's in the kit rows), untouched and uncounted here.
    TRN.Transition.forward = H.transition_forward(prev["transition"], accept=lambda mod, x, cs: x.shape[-1] == 64)
    _STATE["patched"] += ["boltz.model.layers.pair_averaging.PairWeightedAveraging.forward", "boltz.model.layers.outer_product_mean.OuterProductMean.forward",
                          "boltz.model.layers.transition.Transition.forward[dim=64]"]
    _STATE["impl"]["hoist"] = H
    return {"state": "on", "reason": None, "supersedes": {k: f"{getattr(v, '__module__', '?')}:{getattr(v, '__qualname__', getattr(v, '__name__', '?'))}" for k, v in prev.items()}}


_T2: Dict[str, Any] = {"calls": 0, "served": 0, "compared": 0, "undetermined": 0, "fallback": 0, "fallback_by": {}, "shapes": {}, "selftest": {}}
_P2: Dict[str, Any] = {"calls": 0, "served": 0, "compared": 0, "undetermined": 0, "fallback": 0, "fallback_by": {}, "classes": {}, "selftest_ms": 0.0}
# The exact claim's scope, per unit: the cards + stack the bit-equality probes ran on. Outside it the exact unit refuses BY NAME at apply (state=skipped
# reason=pin:...); inside it every shape class is bit-compared against the stock statements under the LOCK DISCIPLINE below. trans2x's single
# replica proves on 8.0 as on 9.0 (A100-80GB: classes rows x float32:0 and :32 proven under the lock at 400 / 800 / 1200 tokens, served 118 of 120
# calls, every output file of --mode exact byte-identical); pwa2x's candidate set is cuBLAS's sm_90 structures — on 8.0 it proves one class in
# three (800x7382:c; bitcmp_failed:400x7311:c, :1200x7410:c: the lock returns stock, correct and idle), so pwa2x stays pinned to 9.0 until an
# sm_80 candidate set is probed (msa2_probe on that card).
EXACT_PIN = {"cc": {"pwa2x": ((9, 0),), "trans2x": ((9, 0), (8, 0))}, "torch": "2.12.", "cuda": "13.0"}   # pwa2x's row restates the core cell's own vouch envelope (opt_core.ops.msa_pwa2.VOUCHED_CC / VOUCH_STACK) as this adapter's by-name step-aside
CORE_PWA2 = "opt_core.ops.msa_pwa2"                          # the shared core's exact-replica PairWeightedAveraging cell (unit pwa2x binds it; the trans2 cells stay in forward/msa2)

# LOCK DISCIPLINE of the exact cells (both units): a first-encounter compare is evidence only if it can discriminate, and only in volume.
#   floor       a class below the unit's size floor is served by the stock statements BY NAME, uncompared (fallback word small_class:<class>; class
#               state 'floor'): the cell buys nothing there and cuBLAS serves tiny problems (a single-sequence MSA, S = 1) with kernels no candidate models.
#   comparing   every call of the class computes the stock statements AND the replica; the call RETURNS THE STOCK OUTPUT. Where the replica selects
#               among candidate summation structures, the class selects candidate C only if C compares torch.equal AND every arithmetically distinct
#               candidate compares NOT equal (candidates are de-duplicated by their k-grouping signature first); all-equal / several-equal = the call is
#               'undetermined' (counted; stock output returned; selection re-attempted at the next call). First mismatch of a selected candidate (or of
#               the single replica) EVER = the class is refused by name for the rest of the process (bitcmp_failed:<class>).
#   proven      once the compares have covered >= LOCK_ELEMS output elements over >= LOCK_CALLS calls with zero mismatches, the replica alone serves.
# Nothing a replica computed is ever returned uncompared below the budget. Under CUDA-graph capture a class that is not yet proven is served by the
# stock statements (capture_unverified) — the compare needs eager execution.
PWA_FLOOR_S = 16                  # MSA rows
PWA_FLOOR_ROWS = 65_536           # S * N; above the floor one PWA call already yields >= 4.19e6 output elements (S*N*64)
T2_FLOOR_ROWS = 65_536            # rows of the dim-64 transition input (S*N for the MSA transition, T*N*N for the template pairformer's)
LOCK_ELEMS = 1_000_000            # GEMM-class replicas: output elements compared equal before a class is served alone
LOCK_CALLS = 1


class _Lock:
    """Per-unit class ledger of the lock discipline. Pure Python (no torch): the units feed it compare outcomes, it returns verdicts."""

    def __init__(self, lock_elems: int = LOCK_ELEMS, lock_calls: int = LOCK_CALLS):
        self.lock_elems, self.lock_calls = lock_elems, lock_calls
        self.classes: Dict[Any, Dict[str, Any]] = {}

    def get(self, key) -> Dict[str, Any]:
        st = self.classes.get(key)
        if st is None:
            st = self.classes[key] = {"state": "comparing", "sel": None, "candidates_tested": 0, "undetermined": 0, "compared_calls": 0, "compared_elems": 0, "ms": 0.0}
        return st

    def floor(self, key) -> None:
        st = self.get(key); st["state"] = "floor"; st["compared_calls"] += 0

    def first(self, key, equal_by_sig: Dict[Any, bool], elems: int, ms: float = 0.0) -> str:
        """Outcome of a selecting compare (the class has no selection yet): equal_by_sig maps each DISTINCT candidate signature -> torch.equal result.
        Returns 'selected' (exactly one equal, others not) | 'undetermined' | 'refused' (none equal)."""
        st = self.get(key); st["ms"] += ms
        if st["state"] != "comparing":                       # a closed class (proven / refused / floor) is never re-opened
            return st["state"]
        st["candidates_tested"] = max(st["candidates_tested"], len(equal_by_sig))
        eq = [sig for sig, ok in equal_by_sig.items() if ok]
        if not eq:
            st["state"] = "refused"; return "refused"
        if len(eq) > 1:
            st["undetermined"] += 1; return "undetermined"
        st["sel"] = eq[0]; st["compared_calls"] += 1; st["compared_elems"] += int(elems)
        self._maybe_proven(st); return "selected"

    def next(self, key, equal: bool, elems: int, ms: float = 0.0) -> str:
        """Outcome of a confirming compare of the selected candidate / single replica: 'ok' | 'refused'."""
        st = self.get(key); st["ms"] += ms
        if st["state"] != "comparing":
            return st["state"]
        if not equal:
            st["state"] = "refused"; return "refused"
        st["compared_calls"] += 1; st["compared_elems"] += int(elems); self._maybe_proven(st); return "ok"

    def _maybe_proven(self, st) -> None:
        if st["state"] == "comparing" and st["compared_elems"] >= self.lock_elems and st["compared_calls"] >= self.lock_calls:
            st["state"] = "proven"

    def count(self, state: str) -> int:
        return sum(1 for st in self.classes.values() if st["state"] == state)

    def words(self, keyfmt) -> str:
        return ";".join(f"{keyfmt(k)}=" + (f"{st['sel']}" if st["sel"] is not None else "-").replace(" ", "") +
                        f":{st['state']}:cand{st['candidates_tested']}/und{st['undetermined']}/calls{st['compared_calls']}/elems{st['compared_elems']}@{st['ms']:.0f}ms"
                        for k, st in self.classes.items()) or "-"

    def census(self, keyfmt) -> Dict[str, Any]:
        return {keyfmt(k): {**st, "sel": (list(st["sel"]) if isinstance(st["sel"], tuple) else st["sel"])} for k, st in self.classes.items()}


_T2_LOCK = _Lock(); _P2_LOCK = _Lock()
_T2["selftest"] = _T2_LOCK.classes; _P2["classes"] = _P2_LOCK.classes


def _t2_key_word(key) -> str:
    return f"{key[0]}x{key[1]}:{key[2]}"


def _cls_word(key) -> str:
    return f"{key[0]}x{key[1]}:{'c' if key[2] else 'u'}"


def _pin_word(unit: str):
    """The exact unit's refusal word outside its pinned scope (pin:no_cuda | pin:cc<MMm> | pin:torch<v> | pin:cuda<v>), None inside it."""
    import torch
    if not torch.cuda.is_available():
        return "pin:no_cuda"
    cc = tuple(torch.cuda.get_device_capability(0))
    if cc not in EXACT_PIN["cc"][unit]:
        return f"pin:cc{cc[0]}{cc[1]}"
    if not torch.__version__.startswith(EXACT_PIN["torch"]):
        return f"pin:torch{torch.__version__.split('+')[0]}"
    if (torch.version.cuda or "") != EXACT_PIN["cuda"]:
        return f"pin:cuda{torch.version.cuda}"
    return None


def _apply_trans2(mode: int = 0) -> Dict[str, Any]:
    import time
    import torch
    _load_package()
    T2 = importlib.import_module(f"{PACKAGE}.trans2")
    from boltz.model.layers import transition as TRN
    prev = TRN.Transition.forward
    if mode == 1:
        word = _pin_word("trans2x")
        if word is not None:                                                 # the exact claim is card/stack-scoped: refuse by name, stock statements serve
            return {"state": "skipped", "reason": word, "mode": "exact"}
    L = _T2_LOCK

    def forward(self, x, chunk_size=None):
        if x.shape[-1] != 64 or self.fc1.weight.shape[0] != 256:          # not this lever's instance: whatever served before, untouched and uncounted
            return prev(self, x, chunk_size)
        _T2["calls"] += 1
        word = None
        if self.training: word = "training"
        elif not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16: word = "autocast_off"
        else: word = T2.supported(self, x, chunk_size)
        if word is None and mode == 1:                                       # the lock discipline (single replica: no selection, budget + floor apply)
            rows = x.numel() // 64
            key = (rows, str(x.dtype).split(".")[-1], int(chunk_size or 0))
            if rows < T2_FLOOR_ROWS:
                L.floor(key); word = f"small_class:{_t2_key_word(key)}"
            else:
                st = L.get(key)
                if st["state"] == "comparing":
                    if torch.cuda.is_current_stream_capturing():
                        word = "capture_unverified"
                    else:
                        t0 = time.perf_counter()
                        ref = prev(self, x, chunk_size)                          # the stock statements: what this call returns
                        got = T2.transition64(self, x, chunk_size, mode=1)
                        eq = bool(got.dtype == ref.dtype and got.shape == ref.shape and torch.equal(got, ref)); del got
                        torch.cuda.synchronize()
                        if L.next(key, eq, ref.numel(), (time.perf_counter() - t0) * 1e3) == "refused":
                            sys.stderr.write(f"[boltz2-opt] MSA2 trans2x BIT-COMPARE FAILED class={_t2_key_word(key)} on compared call {st['compared_calls'] + 1}: the stock statements serve this class for the process\n")
                        _T2["compared"] += 1
                        return ref
                if st["state"] == "refused":
                    word = f"bitcmp_failed:{_t2_key_word(key)}"
        if word is not None:
            _T2["fallback"] += 1; _T2["fallback_by"][word] = _T2["fallback_by"].get(word, 0) + 1
            return prev(self, x, chunk_size)
        _T2["served"] += 1
        k = f"{x.numel() // 64}x64:{chunk_size}"; _T2["shapes"][k] = _T2["shapes"].get(k, 0) + 1
        return T2.transition64(self, x, chunk_size, mode=mode)
    forward.__wrapped__ = prev; forward._msa2 = "trans2x" if mode == 1 else "trans2"
    TRN.Transition.forward = forward
    _STATE["patched"].append("boltz.model.layers.transition.Transition.forward[dim=64,hidden=256]")
    _STATE["impl"]["trans2"] = T2
    out = {"state": "on", "reason": None, "mode": "exact" if mode == 1 else "fast", "supersedes": f"{getattr(prev, '__module__', '?')}:{getattr(prev, '__qualname__', '?')}"}
    if mode == 1:
        out.update({"floor": f"rows>={T2_FLOOR_ROWS}", "lock": f"elems>={LOCK_ELEMS},calls>={LOCK_CALLS}"})
    return out


def _apply_pwa2x() -> Dict[str, Any]:
    import time
    import torch
    P2 = importlib.import_module(CORE_PWA2)                          # the shared core's exact-replica cell; this adapter owns the Boltz-2 binding, the pin and the lock
    from boltz.model.layers import pair_averaging as PA
    prev = PA.PairWeightedAveraging.forward
    word = _pin_word("pwa2x")
    if word is not None:
        return {"state": "skipped", "reason": word, "mode": "exact"}
    L = _P2_LOCK

    def forward(self, m, z, mask, chunk_heads=False):
        _P2["calls"] += 1
        word = None
        if self.training: word = "training"
        elif not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16: word = "autocast_off"
        else: word = P2.supported(self, m, z, mask, chunk_heads)
        if word is None:
            N, S = int(m.shape[2]), int(m.shape[1])
            key = (N, S, bool(chunk_heads and not self.training))
            if S < PWA_FLOOR_S or S * N < PWA_FLOOR_ROWS:
                L.floor(key); word = f"small_class:{_cls_word(key)}"
            else:
                st = L.get(key)
                if st["state"] == "proven":
                    _P2["served"] += 1
                    return P2.pwa_forward(self, m, z, mask, chunk_heads, st["sel"][0], st["sel"][1])
                if st["state"] == "comparing":
                    if torch.cuda.is_current_stream_capturing():
                        word = "capture_unverified"
                    else:
                        t0 = time.perf_counter()
                        ref = prev(self, m, z, mask, chunk_heads)                    # the stock statements: what this call returns
                        if st["sel"] is None:                                        # selecting compare: every DISTINCT candidate, exactly one must match
                            eq = {}
                            for km, R in P2.distinct_candidates(N):
                                cand = P2.pwa_forward(self, m, z, mask, chunk_heads, km, R)
                                eq[(km, R)] = bool(cand.dtype == ref.dtype and cand.shape == ref.shape and torch.equal(cand, ref)); del cand
                            torch.cuda.synchronize()
                            v = L.first(key, eq, ref.numel(), (time.perf_counter() - t0) * 1e3)
                            if v == "refused":
                                sys.stderr.write(f"[boltz2-opt] MSA2 pwa2x BIT-COMPARE: no candidate summation structure reproduces cuBLAS for class {_cls_word(key)} ({eq}): the stock forward serves it for the process\n")
                            elif v == "undetermined":
                                _P2["undetermined"] += 1
                                w_ = f"undetermined:{_cls_word(key)}"; _P2["fallback_by"][w_] = _P2["fallback_by"].get(w_, 0) + 1
                        else:                                                        # confirming compare of the selected candidate until the budget is met
                            cand = P2.pwa_forward(self, m, z, mask, chunk_heads, st["sel"][0], st["sel"][1])
                            eq1 = bool(torch.equal(cand, ref)); del cand
                            torch.cuda.synchronize()
                            if L.next(key, eq1, ref.numel(), (time.perf_counter() - t0) * 1e3) == "refused":
                                sys.stderr.write(f"[boltz2-opt] MSA2 pwa2x BIT-COMPARE: class {_cls_word(key)} candidate {st['sel']} mismatched on compared call {st['compared_calls'] + 1}: the stock forward serves it for the process\n")
                        _P2["selftest_ms"] = sum(c["ms"] for c in L.classes.values())
                        _P2["compared"] += 1                                         # a compared call returned the stock output (bitwise by identity)
                        return ref
                if st["state"] == "refused":
                    word = f"bitcmp_failed:{_cls_word(key)}"
        _P2["fallback"] += 1; _P2["fallback_by"][word] = _P2["fallback_by"].get(word, 0) + 1
        return prev(self, m, z, mask, chunk_heads)
    forward.__wrapped__ = prev; forward._msa2 = "pwa2x"
    PA.PairWeightedAveraging.forward = forward
    _STATE["patched"].append("boltz.model.layers.pair_averaging.PairWeightedAveraging.forward")
    _STATE["impl"]["pwa2"] = P2
    return {"state": "on", "reason": None, "mode": "exact", "supersedes": f"{getattr(prev, '__module__', '?')}:{getattr(prev, '__qualname__', '?')}",
            "floor": f"S>={PWA_FLOOR_S},S*N>={PWA_FLOOR_ROWS}", "lock": f"elems>={LOCK_ELEMS},calls>={LOCK_CALLS}"}


def _lever_lines() -> List[str]:
    out = []
    st = _STATE["state"].get("pwa2x")
    if st is not None:
        fb = ",".join(f"{w}:{n}" for w, n in _P2["fallback_by"].items()) or "none"
        L = _P2_LOCK
        out.append(f"[boltz2-opt] LEVER name=msa_pwa_exact state={st['state']} tier=1 numerics=exact impl=opt_core.ops.msa_pwa2:pwa_forward origin=core strategy=LOCAL.boltz2.fused_pwa "
                   f"mode=exact served={_P2['served']} compared={_P2['compared']} undetermined={_P2['undetermined']} calls={_P2['calls']} fallback={_P2['fallback']} fallback_by={fb} "
                   f"selftest=classes:{len(L.classes)},proven:{L.count('proven')},comparing:{L.count('comparing')},refused:{L.count('refused')},floor:{L.count('floor')} "
                   f"selection_ms={_P2['selftest_ms']:.0f} floor=S>={PWA_FLOOR_S},SxN>={PWA_FLOOR_ROWS} lock=elems>={LOCK_ELEMS} classes={L.words(_cls_word)}"
                   + (f" reason={st['reason']}" if st.get("reason") else ""))
    for unit, lever, tier, cls in (("trans2", "msa_trans2", 2, "fast"), ("trans2x", "msa_trans2_exact", 1, "exact")):
        st = _STATE["state"].get(unit)
        if st is None:
            continue
        fb = ",".join(f"{w}:{n}" for w, n in _T2["fallback_by"].items()) or "none"
        L = _T2_LOCK
        scw = (f" compared={_T2['compared']} selftest=classes:{len(L.classes)},proven:{L.count('proven')},comparing:{L.count('comparing')},refused:{L.count('refused')},floor:{L.count('floor')} "
               f"floor=rows>={T2_FLOOR_ROWS} lock=elems>={LOCK_ELEMS} classes={L.words(_t2_key_word)}") if unit == "trans2x" else ""
        out.append(f"[boltz2-opt] LEVER name={lever} state={st['state']} tier={tier} numerics={cls} impl=forward/msa2/trans2.py:transition64 origin=kit "
                   f"strategy=LOCAL.boltz2.fused_transition64 mode={st.get('mode')} served={_T2['served']} calls={_T2['calls']} fallback={_T2['fallback']} fallback_by={fb}{scw}"
                   + (f" reason={st['reason']}" if st.get("reason") else ""))
    return out


def _at_exit():
    try:
        for ln in _lever_lines():
            sys.stderr.write(ln + "\n")
        sys.stderr.flush()
    except Exception as e:  # never mask the worker's exit code
        sys.stderr.write(f"[boltz2-opt] msa2: lever line not written: {type(e).__name__}: {e}\n")


def apply() -> List[str]:
    req = requested()
    _STATE["requested"] = req
    bad = [u for u in req if u not in UNITS]
    if bad:
        raise RuntimeError(f"{SWITCH}: unknown unit(s) {bad} (units: {','.join(UNITS)})")
    applied = []
    if "hoist" in req:
        _STATE["state"]["hoist"] = _apply_hoist(); applied.append("hoist")
    if "trans2" in req and "trans2x" in req:
        raise RuntimeError(f"{SWITCH}: units trans2 (fast class) and trans2x (exact class) are mutually exclusive")
    if "trans2" in req:                                  # after hoist: takes the dim-64 transitions from it (fast class), hoist keeps PWA/OPM
        _STATE["state"]["trans2"] = _apply_trans2(0); applied.append("trans2")
    if "pwa2x" in req:                                   # after hoist (takes PWA from it; hoist keeps OPM)
        _STATE["state"]["pwa2x"] = _apply_pwa2x(); applied.append("pwa2x")
    if "trans2x" in req:
        _STATE["state"]["trans2x"] = _apply_trans2(1); applied.append("trans2x")
    if "probe" in req:                                   # last: its leaf timers wrap whatever serves (installed lazily at the first forward anyway)
        from . import msa2_probe
        msa2_probe.install()
        _STATE["state"]["probe"] = {"state": "on", "reason": None}
        applied.append("probe")
    _STATE["applied"] = applied
    atexit.register(_at_exit)
    sys.stderr.write(f"[boltz2-opt] MSA2 units={','.join(applied) or '-'} requested={','.join(req) or '-'} levers={','.join(UNIT_LEVER[u] for u in applied if u in UNIT_LEVER) or '-'}\n")
    return applied


def report() -> Dict[str, Any]:
    out: Dict[str, Any] = {"applied": list(_STATE["applied"]), "requested": list(_STATE["requested"]), "units": dict(_STATE["state"]),
                           "patched": list(_STATE["patched"]), "lines": _lever_lines()}
    H = _STATE["impl"].get("hoist")
    if H is not None:
        out["hoist_census"] = H.census()
    if "trans2" in _STATE["applied"] or "trans2x" in _STATE["applied"]:
        out["trans2_census"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _T2.items() if k != "selftest"}
        out["trans2_census"]["selftest"] = _T2_LOCK.census(_t2_key_word)
        out["trans2_census"]["floor"] = {"rows": T2_FLOOR_ROWS}; out["trans2_census"]["lock"] = {"elems": LOCK_ELEMS, "calls": LOCK_CALLS}
    if "pwa2x" in _STATE["applied"]:
        out["pwa2_census"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _P2.items() if k != "classes"}
        out["pwa2_census"]["classes"] = _P2_LOCK.census(_cls_word)
        out["pwa2_census"]["floor"] = {"S": PWA_FLOOR_S, "rows": PWA_FLOOR_ROWS}; out["pwa2_census"]["lock"] = {"elems": LOCK_ELEMS, "calls": LOCK_CALLS}
    if "probe" in _STATE["applied"]:
        from . import msa2_probe
        out["probe"] = msa2_probe.report()
    return out
