"""Evidence of the kit arm's levers, read ONLY from the lines the arm printed in its own process — one `[colabdesign-opt] LEVER name=<id>
state=<on|skipped|off> [reason=<why>] impl=… origin=… k=v …` line per lever (`opt_core.report.lever_line`: nosub.py prints its line at model
build, pallas.py at process exit with the Ledger's census). Per lever of the mode one of STATES:

* ``applied``  — `state=on` (nosub: the design step's program is unchunked; pallas: `served` > 0 with only DECLARED fallback reasons);
* ``gated``    — nosub / transition `state=skipped reason=gated`: at or below the lever's size gate the programs are the mode's without it by construction
                  (not a failure);
* ``fallback`` — pallas with a fallback reason outside the declared set; nosub with any word but `state=on` / `reason=gated` (the lever has
                  no fallback of its own: such a line is a defect, named rather than passed);
* ``missing``  — no line of its own (the lever never ran, or the process did not exit normally), or pallas with `served=0`;
* ``skipped``  — `state=skipped reason=cannot_run|no_attention_kernel [detail=…]`: the lever stepped aside BY NAME at install (levers.py —
                  the kit rule: a lever that cannot engage here never refuses the mode and never disappears silently); or, at run time, EVERY
                  call of a census lever stepped aside by a DECLARED word (served=0, fallback_by=<declared>:n — e.g. trimul on a configuration
                  without fused projection weights, opm_fold gated_s_gt_32, hoist_prev on the multi-model path): skipped_by says
                  `aside:<word>:<n>`. Not a failure, exit 0, named on the EVIDENCE line's `skipped=` and by `cli.partial_exit` with its reason.

``verdict`` turns the record into the exit rule's inputs: ``partial`` (fallback / missing of an INSTALLED lever → a defect, exit 3), ``gated``
and ``skipped`` (each {lever: reason}, exit 0).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

from . import registry
from .registry import ORDER
from .names import LEVER_COMPILECACHE, LEVER_LOWERCACHE, LEVER_HOIST_PREV, LEVER_PARCOMPILE, LEVER_NOSUB, LEVER_NOSUB_FN, LEVER_PALLAS, LEVER_TRIATT, LEVER_TRIMUL, LEVER_OPM_FOLD, LEVER_LN, LEVER_PROJ, LEVER_TXLA, LEVER_TRANSITION
from .pallas import EXPECTED_FALLBACKS as DECLARED_FALLBACKS      # the ONE declaration (pallas.py, from the core's reason names), as printed on the LEVER line
from .levers import STEP_ASIDE_REASONS                            # the reason words of a lever that stepped aside by name (state=skipped): cannot_run | no_attention_kernel | no_gpu

LINE_NAMES = registry.line_names()                                                  # {lever: the name= token of its LEVER line} (registry.py)
TRIMUL_LINE_NAME = LINE_NAMES[LEVER_TRIMUL]                                          # kernels/trimul_fused.py LINE_NAME
TRIMUL_DECLARED_FALLBACKS = ("not_fused", "shape", "dtype", "equation")                                           # kernels/trimul_fused.EXPECTED_FALLBACKS (locked by test): stock's own program for a config without fused projection weights
TRIATT_LINE_NAME = LEVER_TRIATT                                                        # kernels/triatt_lever.py: the LEVER line's name= is the lever id
TRIATT_DECLARED_FALLBACKS = ("below_keys_rule", "no_pair_bias", "key_dim_ne_value_dim", "bias_form", "below_size_rule", "head_dim_lt_16")                                     # kernels/triatt_lever.EXPECTED_FALLBACKS: calls with fewer than 16 keys run stock's method (bitwise), counted
OPM_FOLD_LINE_NAME, OPM_FOLD_DECLARED_FALLBACKS = LEVER_OPM_FOLD, ("gated_s_gt_32",)      # kernels/layers_opm.py NAME / EXPECTED_FALLBACKS
LN_LINE_NAME, LN_DECLARED_FALLBACKS = LEVER_LN, ("channels_not_pow2", "channels_range", "rank", "axis", "param_axis", "dtype")                      # kernels/layers_ln.py NAME / EXPECTED_FALLBACKS
PROJ_LINE_NAME, PROJ_DECLARED_FALLBACKS = LEVER_PROJ, ("head_dim_lt_16", "no_gating", "no_pair_bias", "key_dim_ne_value_dim", "bias_form", "below_keys_rule", "below_size_rule")                     # kernels/proj_attn.py LEVER / EXPECTED_FALLBACKS
TRANSITION_LINE_NAME, TRANSITION_DECLARED_FALLBACKS = LEVER_TRANSITION, ("dtype_not_bf16", "channels", "intermediate_width", "rank")      # kernels/layers_transition.py NAME / EXPECTED_FALLBACKS (locked by test): the size rule hands the call to the stock class BY NAME
TXLA_LINE_NAME, TXLA_DECLARED_FALLBACKS = LEVER_TXLA, ("small_call", "refused")   # txla.py TAG / EXPECTED_FALLBACKS (locked by test): each hands the call to triatt BY NAME
GATED_REASON = "gated"                                                               # a census lever's `state=skipped reason=gated` (nosub's word): the programs are the
                                                                                     # mode's without the lever by construction — classified `gated` (exit 0), never missing
# the census levers, registry order: lever id -> (its LEVER line's name= token, the fallback reasons its module declares). One classification rule
# for all of them (classify) and one pair of EVIDENCE fields each (`<id>_served=` `<id>_fallback_by=`, report.evidence_line).
CELL_FALLBACK_PREFIX = "cell_"                                                             # kernels/provider.CELL_FALLBACK (locked by test): a call the provider handed to the stock method by its measured cell (`cell_xla`, ...) — a per-call step-aside BY NAME


def _declared(word: str, declared) -> bool:
    """A per-call fallback word is DECLARED when it is one of the lever's structural step-aside words or a provider cell word."""
    return word in declared or word.startswith(CELL_FALLBACK_PREFIX)


CENSUS_LEVERS = {LEVER_TRIMUL: (TRIMUL_LINE_NAME, TRIMUL_DECLARED_FALLBACKS), LEVER_TRIATT: (TRIATT_LINE_NAME, TRIATT_DECLARED_FALLBACKS),
                 LEVER_OPM_FOLD: (OPM_FOLD_LINE_NAME, OPM_FOLD_DECLARED_FALLBACKS), LEVER_LN: (LN_LINE_NAME, LN_DECLARED_FALLBACKS),
                 LEVER_PROJ: (PROJ_LINE_NAME, PROJ_DECLARED_FALLBACKS), LEVER_TRANSITION: (TRANSITION_LINE_NAME, TRANSITION_DECLARED_FALLBACKS),
                 LEVER_TXLA: (TXLA_LINE_NAME, TXLA_DECLARED_FALLBACKS)}
PALLAS_LINE_NAME = LINE_NAMES[LEVER_PALLAS]                                          # pallas.py's Ledger name on the line (opt_core.kernels.pallas_attn_serve.LEVER)
ABLATED_REASON = "ablated"                                                           # the reason word of a dropped lever's `state=off` line (levers.install)
RE_LEVER = re.compile(r"\[colabdesign-opt\] LEVER (name=\S+ state=\S+.*)$")
RE_KV = re.compile(r"(\S+?)=(\S+)")
STATES = ("applied", "gated", "fallback", "missing", "skipped")


def lever_lines(lines: Iterable[str]) -> List[dict]:
    """Every LEVER line of the arm, parsed to {"name", "state", ...k: v} (values as printed: strings)."""
    out = []
    for raw in lines:
        m = RE_LEVER.search(str(raw).rstrip("\n"))
        if m:
            d = dict(RE_KV.findall(m.group(1)))
            d["_line"] = m.group(0)
            out.append(d)
    return out


def _fallback_by(text: Optional[str]) -> Dict[str, int]:
    """`reason:n,reason:n` → {reason: n} ({} for None / '{}' / 'none')."""
    if not text or text in ("{}", "none"):
        return {}
    out = {}
    for part in text.strip("{}").split(","):
        if ":" in part:
            k, v = part.rsplit(":", 1)
            try:
                out[k.strip().strip("'\" ")] = int(v)
            except ValueError:
                pass
    return out


def _int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def classify(lines: Iterable[str], levers: Iterable[str], ablated: Iterable[str] = (), restored: Iterable[str] = ()) -> Dict[str, object]:
    """The evidence record for a kit arm: per lever of the mode one of STATES, the values the levers printed, and the lever lines themselves;
    ``ablated`` = the levers a subtractive mode word dropped (modes.resolve): recorded, printed `state=off reason=ablated` by the arm, never classified."""
    lines = list(lines)
    ll = lever_lines(lines)
    by_name = {}
    for d in ll:                                                                    # the LAST line of a name wins (nosub prints once per distinct build)
        by_name[d.get("name")] = d
    want = list(levers)
    rec: Dict[str, object] = {"requested": want, "ablated": list(ablated), "restored": list(restored), "state": {}, "applied": [], "gated": [], "fallback": [], "missing": [], "skipped": [], "skipped_by": {},
                              "tokens": None, "grad_subbatch": None, "grad_subbatch_source": None, "fn_subbatch": None, "fn_subbatch_source": None,
                              "triatt_served": None, "triatt_fallback": None, "trimul_served": None, "trimul_fallback": None, "trimul_fallback_by": None, "trimul_impl": None, "trimul_precision": None,
                              "pallas_served": None, "pallas_fallback": None, "pallas_fallback_by": None, "pallas_impl": None, "pallas_shapes": None,
                              "compile_requests": None, "compile_hits": None, "lowercache_calls": None, "lowercache_loads": None, "lowercache_stores": None, "parcompile_threads": None, "compile_misses": None, "compile_dir_source": None, "prev_device_steps": None, "prev_stock_path": None,
                              "lever_lines": [d["_line"] for d in ll], "off": sorted(d.get("name") for d in ll if d.get("state") == "off")}
    aside: Dict[str, str] = {}                                                      # the levers that stepped aside BY NAME at install (levers.py): a `state=skipped reason=<STEP_ASIDE_REASONS>` line of
    for lever in want:                                                              # the lever's name and no later `state=on` line of it — classified `skipped` (exit 0), never applied / missing
        mine = [d for d in ll if d.get("name") == LINE_NAMES.get(lever, lever)]
        stepped = [d for d in mine if d.get("state") == "skipped" and d.get("reason") in STEP_ASIDE_REASONS]
        if stepped and mine[-1].get("state") != "on":
            aside[lever] = f"{stepped[-1].get('reason')}: {stepped[-1].get('detail', 'none')}"
    rec["skipped_by"] = dict(aside)
    named_aside: Dict[str, str] = {}                                                 # levers whose EVERY call stepped aside by a declared word at run time (classified below): named, exit 0
    live = [l for l in want if l not in aside]                                       # the levers whose own evidence is read below
    if LEVER_COMPILECACHE in live:                                                  # jax's own cache events counted at exit: a design process always compiles, so requests=0 is missing (fail-closed)
        d = by_name.get(LEVER_COMPILECACHE)
        if d is None or d.get("state") != "on":
            state = "missing"
        else:
            rec["compile_requests"], rec["compile_hits"], rec["compile_misses"] = _int(d.get("requests")), _int(d.get("hits")), _int(d.get("misses"))
            rec["compile_dir_source"] = d.get("dir_source")
            state = "applied" if (rec["compile_requests"] or 0) > 0 else "missing"
        rec["state"][LEVER_COMPILECACHE] = state
    if LEVER_LOWERCACHE in live:                                                    # the exit census: a design process always builds a model, so calls=0 is missing (fail-closed)
        d = by_name.get(LEVER_LOWERCACHE)
        if d is None or d.get("state") != "on":
            state = "missing"
        else:
            rec["lowercache_calls"], rec["lowercache_loads"], rec["lowercache_stores"] = _int(d.get("calls")), _int(d.get("loads")), _int(d.get("stores"))
            state = "applied" if (rec["lowercache_calls"] or 0) > 0 else "missing"
        rec["state"][LEVER_LOWERCACHE] = state
    if LEVER_PARCOMPILE in live:                                          # one line at install: the flags in force from there (threads=<n>); nothing later can change them
        d = by_name.get(LEVER_PARCOMPILE)
        if d is None or d.get("state") != "on":
            state = "missing"
        else:
            rec["parcompile_threads"] = _int(d.get("threads"))
            state = "applied"
        rec["state"][LEVER_PARCOMPILE] = state
    if LEVER_HOIST_PREV in live:                                                    # the last line of the name wins (install, then the exit census): the device path taken at least once = applied
        d = by_name.get(LEVER_HOIST_PREV)
        if d is None or d.get("state") != "on":
            state = "missing"
        else:
            rec["prev_device_steps"], rec["prev_stock_path"], runs = _int(d.get("device_prev_steps")), _int(d.get("multi_model_stock_path")), _int(d.get("run_calls"))
            if (rec["prev_device_steps"] or 0) > 0:
                state = "applied"
            elif (runs or 0) > 0 and (rec["prev_stock_path"] or 0) > 0:                    # every design step ran stock's multi-model path BY NAME (a settings file the device path does not apply to): a named aside, exit 0 — never a refusal on a setting stock accepts
                named_aside[LEVER_HOIST_PREV] = f"aside:multi_model_stock_path:{rec['prev_stock_path']} (every design step ran stock's multi-model recycle path by name)"
                state = "skipped"
            else:
                state = "fallback" if (runs or 0) > 0 else "missing"                       # never exercised (no design step ran) = missing, like a kernel that served no call; exercised with neither path counted = fallback (a defect)
        rec["state"][LEVER_HOIST_PREV] = state
    if LEVER_NOSUB in live:
        d = by_name.get(LEVER_NOSUB)
        if d is None or d.get("state") == "off":
            state = "missing"
        else:
            rec["tokens"] = int(d["tokens"]) if str(d.get("tokens", "")).isdigit() else None
            rec["grad_subbatch"], rec["grad_subbatch_source"] = d.get("grad_subbatch"), d.get("grad_subbatch_source")
            rec["fn_subbatch"], rec["fn_subbatch_source"] = d.get("fn_subbatch"), d.get("fn_subbatch_source")
            if d.get("state") == "on":
                state = "applied"
            elif d.get("reason") == "gated":
                state = "gated"
            else:
                state = "fallback"
        rec["state"][LEVER_NOSUB] = state
    if LEVER_NOSUB_FN in live:
        d = by_name.get(LEVER_NOSUB_FN)
        if d is None or d.get("state") == "off":
            state = "missing"
        else:
            if rec["tokens"] is None:
                rec["tokens"] = int(d["tokens"]) if str(d.get("tokens", "")).isdigit() else None
            rec["fn_subbatch"], rec["fn_subbatch_source"] = d.get("fn_subbatch"), d.get("fn_subbatch_source")   # the forward-only executable's decision is this lever's when it is in the set
            if d.get("state") == "on":
                state = "applied"
            elif d.get("reason") == "gated":
                state = "gated"
            else:
                state = "fallback"
        rec["state"][LEVER_NOSUB_FN] = state
    for lever, (line_name, declared) in CENSUS_LEVERS.items():                      # the kit's census levers: applied iff on, served > 0 and every fallback reason declared
        if lever not in live:
            continue
        d = by_name.get(line_name)
        if d is None or d.get("state") == "off":                                          # no line of its own (or another mode's off line): missing
            state = "missing"
        else:                                                                              # on, or skipped (reason=no_calls: served=0 | reason=gated: below the lever's size rule) — the census is recorded either way
            served = int(d["served"]) if str(d.get("served", "")).isdigit() else 0
            fb = _fallback_by(d.get("fallback_by"))
            rec[f"{lever}_served"], rec[f"{lever}_fallback"], rec[f"{lever}_fallback_by"] = served, sum(fb.values()), fb
            rec[f"{lever}_impl"], rec[f"{lever}_precision"] = d.get("impl"), d.get("precision")
            unexpected = {r: v for r, v in fb.items() if not _declared(r, declared)}
            if served == 0 and fb and not unexpected and d.get("state") in ("on", "skipped") and d.get("reason") != GATED_REASON:   # every call of the lever stepped aside by a DECLARED word (trimul on a non-fused / monomer config: not_fused:n; opm_fold: gated_s_gt_32:n; ...): ACTIVE with a NAMED aside, exit 0 — an installed lever that served nothing WITHOUT a declared word stays `missing` (exit 3)
                named_aside[lever] = "aside:" + ",".join(f"{r}:{v}" for r, v in fb.items()) + " (every call stepped aside by a declared word; stock's own path served them)"
                state = "skipped"
            elif d.get("state") == "skipped" and d.get("reason") == GATED_REASON and not unexpected:   # a census lever gated by its size rule (transition): the stock class served by a declared word
                state = "gated"
            else:
                state = "missing" if (served == 0 or d.get("state") != "on") else ("fallback" if unexpected else "applied")
        rec["state"][lever] = state
    if LEVER_PALLAS in live:
        d = by_name.get(PALLAS_LINE_NAME)
        if d is None or d.get("state") == "off":
            state = "missing"
        else:
            served = int(d["served"]) if str(d.get("served", "")).isdigit() else 0
            fb = _fallback_by(d.get("fallback_by"))
            rec["pallas_served"], rec["pallas_fallback"], rec["pallas_fallback_by"] = served, sum(fb.values()), fb
            rec["pallas_impl"], rec["pallas_shapes"] = d.get("impl"), d.get("shapes")
            unexpected = {k: v for k, v in fb.items() if not _declared(k, DECLARED_FALLBACKS)}
            if served == 0 and fb and not unexpected:                                     # every call stepped aside by a declared word: a named aside, exit 0
                named_aside[LEVER_PALLAS] = "aside:" + ",".join(f"{r}:{v}" for r, v in fb.items()) + " (every call stepped aside by a declared word; stock's own path served them)"
                state = "skipped"
            elif served == 0:
                state = "missing"
            elif unexpected:
                state = "fallback"
            else:
                state = "applied"
        rec["state"][LEVER_PALLAS] = state
    for lever in aside:
        rec["state"][lever] = "skipped"
    rec["skipped_by"].update(named_aside)
    for lever in [l for l in want if l in rec["state"]] + [l for l in rec["state"] if l not in want]:   # the run's lever order (registry order), whatever order the branches above ran in
        rec[rec["state"][lever]].append(lever)
    return rec


GATES = {LEVER_NOSUB: "tokens at or below the size gate: stock traces no sub-batch there, the programs are stock's - nothing to apply (modes.SIZE_GATE_TOKENS)",
         LEVER_NOSUB_FN: "tokens at or below the size gate: stock's forward-only executable is unchunked there already - nothing to apply (modes.SIZE_GATE_TOKENS)",
         }


def verdict(ev: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    """{"partial": {lever: reason}, "gated": {lever: reason}, "skipped": {lever: reason}} for a record from ``classify`` (the mode's lever
    order): partial = an installed lever that did not engage (exit 3); gated / skipped = named, exit 0."""
    partial: Dict[str, str] = {}
    gated: Dict[str, str] = {}
    aside: Dict[str, str] = {}
    for lever, state in sorted(ev["state"].items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else len(ORDER)):   # the run's lever order
        if state == "fallback" and lever == LEVER_NOSUB:
            partial[lever] = f"fallback: grad_subbatch={ev.get('grad_subbatch')} source={ev.get('grad_subbatch_source')} (not the shipped decision)"
        elif state == "fallback" and lever == LEVER_HOIST_PREV:
            partial[lever] = f"fallback: installed, the device path never taken (device_prev_steps=0, multi_model_stock_path={ev.get('prev_stock_path')})"
        elif state == "fallback" and lever in CENSUS_LEVERS:
            partial[lever] = f"fallback: undeclared reason(s) {ev.get(lever + '_fallback_by')} (declared: {','.join(CENSUS_LEVERS[lever][1])})"
        elif state == "missing" and lever in CENSUS_LEVERS and ev.get(f"{lever}_served") == 0:
            partial[lever] = "missing: installed, the kernels served no call (served=0)"
        elif state == "fallback":
            partial[lever] = f"fallback: undeclared reason(s) {ev.get('pallas_fallback_by')} (declared: {','.join(DECLARED_FALLBACKS)})"
        elif state == "missing" and lever == LEVER_COMPILECACHE and ev.get("compile_requests") == 0:
            partial[lever] = "missing: installed, no compile request consulted the cache (requests=0)"
        elif state == "missing" and lever == LEVER_HOIST_PREV and ev.get("prev_device_steps") == 0:
            partial[lever] = "missing: installed, no design step ran through it (run_calls=0)"
        elif state == "missing" and lever == LEVER_PALLAS and ev.get("pallas_served") == 0:
            partial[lever] = "missing: installed, no attention call served by the kernel (served=0)"
        elif state == "missing":
            partial[lever] = "missing: no line of its own in the arm's log (the lever did not run, or the process did not exit normally)"
        elif state == "gated":
            gated[lever] = GATES[lever]
        elif state == "skipped":
            aside[lever] = (ev.get("skipped_by") or {}).get(lever) or "stepped aside at install (reason not printed)"
    return {"partial": partial, "gated": gated, "skipped": aside}


def stock_lever_lines(lines: Iterable[str]) -> List[str]:
    """The LEVER lines found in a STOCK arm's log — any is a defect (the stock arm must run no lever)."""
    return [d["_line"] for d in lever_lines(lines)]


def summary(rec: Optional[dict]) -> str:
    if not rec:
        return "n/a"
    return ";".join(f"{k}={v}" for k, v in (rec.get("state") or {}).items()) or "none"
