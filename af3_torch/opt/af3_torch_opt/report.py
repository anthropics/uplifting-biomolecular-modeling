"""The evidence lines and the activation report's text forms. One prefix for every line the package prints."""
from __future__ import annotations

import json
import re
import sys

from typing import List, Optional, Tuple
from opt_core.report import dump, emit, kv, lever_line as _core_lever_line, prefix  # noqa: F401 — dump/emit are the core's (re-exported for cli)

from . import big as _big
from . import registry as _registry

TAG = "af3-torch-opt"
PREFIX = prefix(TAG)
RANKENV_MARK = f"{PREFIX} RANKENV "                 # the shared core launcher's once-per-launch line under this kit's tag (opt_core.mem.rowpair.rankdata.rankenv_line with ROWPAIR_TAG = TAG): `[af3-torch-opt] RANKENV hashseed=<v> source=default|inherited ranks=P`
ACTIVE, NOT_ACTIVE = "ACTIVE", "NOT ACTIVE"


def rowpair_state(rep: dict) -> Tuple[str, Optional[str], dict]:
    """(state, reason, evidence) of the n_gpu axis's lever `rowpair` in one pred: `off reason=n_gpu=1` at P = 1 (no rank, no collective:
    the single-GPU bytes); `on` with rank 0's forward.json `rowpair` record (P, the row alignment, what the ranks ran at each seam, the
    adapter's census counts); `skipped` when P > 1 was asked and the model process left no record (its failure is the run's, named on forward.json)."""
    from . import stack as _stack
    P = int(rep.get("n_gpu") or 1)
    fields = dict(_stack.n_gpu_fields(P))
    if P == 1:
        return "off", "n_gpu=1", fields
    rec = rep.get("rowpair_record") or {}
    if not rec:
        return "skipped", "no_rowpair_record(forward.json)", fields
    ev = dict(fields, ranks=rec.get("P"), align=rec.get("align"), trimul=rec.get("trimul"), triattn=rec.get("triattn"), heads=rec.get("heads"),
              diffusion=rec.get("diffusion"), dtk=rec.get("dtk"), core=rec.get("core_version"), prev_free=rec.get("prev_free") or 0,
              **{k: v for k, v in (rec.get("stats") or {}).items() if k in ("items", "pair_blocks", "presharded_calls", "gathers_in_trunk", "conf_rows", "zcond_rows", "dit_blocks")})
    return "on", None, ev


def n_gpu_text(rep: dict) -> str:
    """`n_gpu=P sharding=rowpair|none` — the resource axis of big on the ACTIVE and DONE lines, in the core's words (stack.n_gpu_fields:
    opt_core.mem.ngpu is the one producer of the token text)."""
    from . import stack as _stack                                      # stack imports this module for its lines: resolved at call time
    return kv(*_stack.n_gpu_fields(int(rep.get("n_gpu") or 1)))


def compile_word(rep: dict) -> str:
    """The ACTIVE / LEVER line's ``compile=`` word: ``on`` = the mode's torch.compile lever is in the selection that runs; ``off:user`` = the user
    switched it off for this run (``--no-compile``, i.e. ``MODEL_OPT_LEVERS_OFF`` naming ``compile`` — one mechanism: cli.no_compile_alias);
    ``off:mode`` = the mode never compiles (``off`` / ``exact``); ``stepped_aside:<reason>`` = selected but a compiled callable raised in the model
    process and its eager statement served (forward._runtime_dead's word, known after the pass: the LEVER line only)."""
    dead = (rep.get("dead") or {}).get("compile") if isinstance(rep.get("dead"), dict) else None
    if dead:
        return "stepped_aside:" + re.sub(r"[^A-Za-z0-9_.+-]", "", str(dead).split(":")[0].strip().replace(" ", "_"))
    if "compile" in (rep.get("levers") or ()):
        return "on"
    if "compile" in (rep.get("levers_off") or ()):
        return "off:user"
    return "off:mode"


def activation_line(rep: dict) -> str:
    """`[af3-torch-opt] ACTIVE mode=fast lever_set=fast levers=bf16w+trimul+… dtk=1 n_gpu=1 sharding=none params=<dir>
    weights=<OpenFold3-preview2|file> sha256=<12> (pinned|unpinned) torch_py=… jax_py=… cache=… gpu_target=…
    package=template_dedupe,dev_scalars,tri_layout,ln_rows,attn_layout,gate_fuse,canonical_noise padding=kernel_tile` or `[af3-torch-opt] NOT ACTIVE mode=… n_gpu=… reason=…` (pred exits 3 on the latter)."""
    if rep.get("active"):
        parts = [PREFIX, ACTIVE, f"mode={rep['mode']}", f"lever_set={rep.get('lever_set')}", f"levers={rep.get('levers_label')}",
                 f"dtk={int(bool(rep.get('dtk')))}", n_gpu_text(rep), f"params={rep.get('params_dir')}",
                 f"weights={rep.get('weights_word')}",
                 f"torch_py={rep.get('torch_python')}", f"jax_py={rep.get('jax_python')}", f"cache={rep.get('cache_root')}", f"gpu_target={rep.get('gpu_target')}"]
        parts += [f"package={','.join(rep.get('package_levers') or ()) or 'none'}", f"padding={rep.get('padding')}"]      # append-only: fields up to gpu_target are a stable line prefix, never reordered
        parts.append(f"compile={compile_word(rep)}")                                    # the torch.compile lever's state for this run: on | off:user (--no-compile) | off:mode (off / exact never compile)
        if rep.get("levers_off"):                                                    # the ablation switch (modes.ENV_LEVERS_OFF) dropped these from the mode for this run: named only then, last
            parts.append(f"levers_off={','.join(rep['levers_off'])}")
        return " ".join(parts)
    return f"{PREFIX} {NOT_ACTIVE} mode={rep.get('mode')} n_gpu={rep.get('n_gpu')} reason={rep.get('reason')}"


def line(tag: str, **kv) -> str:
    return " ".join([PREFIX, tag] + [f"{k}={v}" for k, v in kv.items()])


PHASES = ("lm", "trunk", "sampler", "conf")        # the PHASE line's per-item phase walls, in order: protein-LM encode (none in AF3: `-`), the trunk (target-feat
                                                   # embedding + every Evoformer recycle), the diffusion sampler (every step of every sample), the confidence head (every sample, summed)


def phase_fields(phase_s: dict, total_s) -> dict:
    """The PHASE line's fields for one (input, seed) — `PHASE item=<name> seed=<s> lm_s=<f|-> trunk_s=<f> sampler_s=<f> conf_s=<f> total_s=<f>`:
    one `<phase>_s` per PHASES from the model process's `phase_s` record (forward.py `_forward_samples`: torch.cuda.synchronize() + perf_counter
    at each call boundary; `-` = the engine has no such phase), and `total_s` = the ITEM line's `forward_s` (the existing per-item timer; the
    phases are sub-intervals of it — the distogram head, big's diff_free eviction and host copies are in total_s and in no phase — so they sum to at most it)."""
    return {**{k + "_s": ("-" if phase_s.get(k) is None else phase_s[k]) for k in PHASES}, "total_s": total_s}


def arch_fields(census) -> dict:
    """`arch=<cc> cells=tuned|builtin` from the kit's census `arch` block (kernels/af3_kernels.py arch_info: the device's compute capability and
    whether its per-arch launch cells were found — `builtin`: flash_triattn's built-in defaults, the kit's FALLBACK CONFIG notice)."""
    a = (census or {}).get("arch") if isinstance(census, dict) else None
    if not isinstance(a, dict):
        return {}
    notices = a.get("notices") or []
    return {"arch": a.get("cc"), "cells": "builtin" if any("FALLBACK CONFIG" in str(n) for n in notices) else "tuned"}


def kernels_line(routes: dict, names, census=None) -> str:
    """`[af3-torch-opt] KERNELS routed=fpf_trimul_v4,flash_triattn ok=1 fpf_trimul_v4=4.1.0:core flash_triattn=1:none arch=9.0 cells=builtin`: the kernels
    the model process served from the shared core's carried copies (registry.KERNEL_ROUTES) — `ok`: every expected route passed
    opt_core.kernels.route_check in the model process (0: refused, rc 2, the reason on forward.json; or the report names no route); per kernel
    `<sums version>:<imported>` — `core` (the module in the process came from the core copy), `none` (no lever imported it: the eager set),
    or `OTHER:<path>` (a defect: the route was bypassed); `arch= cells=`: arch_fields (the model process's device and whether the kit's
    tuned launch cells cover it)."""
    names = list(names)
    if not routes:
        return f"{PREFIX} KERNELS " + kv(routed=None, expected=names, ok=0, **arch_fields(census))
    per, bypassed = {}, []
    for name, rec in routes.items():
        src, core = rec.get("imported_from"), rec.get("core_copy")
        where = "none" if not src else ("core" if core and str(src).startswith(str(core).rstrip("/")) else f"OTHER:{src}")
        if where.startswith("OTHER"):
            bypassed.append(name)                                    # a routed module imported from anywhere but the core copy: the route was bypassed — ok=0
        ver = str(rec.get("version") or "none").split()[0]          # the sums file's version word (its first token: 'as carried (docstring header)' -> 'as')
        per[name] = f"{ver}:{where}"
    ok = int(all(bool(v.get("ok")) for v in routes.values()) and set(routes) == set(names) and not bypassed)
    return f"{PREFIX} KERNELS " + kv(routed=list(routes), ok=ok, **per, **arch_fields(census))


def lever_state(lever: str, rep: dict) -> tuple:
    """(state, reason, evidence) of one registry lever in one pred's records — state per the tree's LEVER-line convention: `on` = applied and
    live in the model process (its own records prove it: registry.EVIDENCE names where — the model's levers record, the kit's per-call census,
    the whole-step graph captures, the compile record, the DTK swap), `off` = not in this mode's lever set, `skipped` = selected by the mode
    but not applied (reason mandatory: the LEVERS / partial census carries the same event)."""
    kind = _registry.EVIDENCE[lever]
    if kind == "package":                                       # a package lever: selected by the mode's package levers, applied per forward.json package_levers_applied, its own census as evidence
        selected = lever in (rep.get("package_levers") or ()); applied = lever in (rep.get("package_levers_applied") or ())
        if not selected and not applied:
            return "off", "not_in_arm_lever_set", {}
        if not applied:
            why = (rep.get("package_levers_skipped") or {}).get(lever)
            return "skipped", (f"skipped:{why}" if why else "absent_from_package_levers_applied(forward.json)"), {}
        c = rep.get(lever) or {}
        return "on", None, {k: v for k, v in c.items() if isinstance(v, (int, float, str))}
    selected = bool(rep.get("dtk")) if kind == "dtk" else lever in (rep.get("levers") or ())
    applied = (rep.get("dtk_swap_s") is not None) if kind == "dtk" else lever in (rep.get("levers_applied") or ())
    dead = lever in (rep.get("dead") or {})                    # a runtime lever that could not run on this stack and stepped aside by name (forward._runtime_dead)
    if not selected and not applied:
        return "off", "not_in_arm_lever_set", {}
    if not applied:
        if kind == "dtk":
            return "skipped", ("dead:swap_failed" if dead else "no_dtk_swap_record(forward.json:dtk_swap_s)"), {}
        return "skipped", "absent_from_levers_applied(forward.json)", {}
    census = rep.get("census") or {}
    counts = census.get(lever) if isinstance(census.get(lever), dict) else {}
    if kind == "census":                                      # the kit's census: {lever: {"served:<key>": n, "fallback:<reason>": n, "kernel_error": n}, on, dead, mask_terms, arch}
        ev = {"served": sum(v for k, v in counts.items() if str(k).startswith("served")), "fallback": sum(v for k, v in counts.items() if str(k).startswith("fallback")),
              "kernel_error": int(counts.get("kernel_error") or 0), "dead": int(lever in (rep.get("dead") or {}))}
    elif kind == "graph_captures":
        ev = {"graph_captures": rep.get("graph_captures")}
    elif kind == "compiled":
        ev = {"compiled": rep.get("compiled")}
    elif kind == "dtk":
        ev = {"swap_s": rep.get("dtk_swap_s")}
    else:
        ev = {"record": "levers_applied"}
    routed = [k for k, r_ in _registry.KERNEL_ROUTES.items() if lever in r_["levers"]]
    if routed:
        ev["impl"] = routed                                    # the module(s) this lever executes, served from the core's carried copies
    if lever == "compile":                                     # the ACTIVE line's compile= word again after the pass: stepped_aside:<reason> when a compiled callable raised and ran eagerly
        ev["compile"] = compile_word(rep)
    if dead and kind != "census":                              # stepgraph / compile / dtk stepped aside at run time: the word the kernel levers carry (their FALLBACK line says dead=1 too); last on the line
        ev["dead"] = 1
    return "on", None, ev


def lever_line(lever: str, rep: dict) -> tuple:
    """(`[af3-torch-opt] LEVER name=F2.trimul state=on arm=fast served=… fallback=… kernel_error=0 dead=0 impl=fpf_trimul_v4`, (state, reason)) —
    one line per registry lever (kit levers and big's memory levers) per pred, in the tree's convention: opt_core.report prefix + kv; name = the
    kit's lever name, strategy = its canonical cross-engine strategy id (registry.STRATEGY); state on|off|skipped with a blank-free reason= when not on."""
    if lever in _big.LEVER_ORDER:
        state, reason, ev = _big.lever_state(lever, rep); fam = _registry.BIG_LEVERS[lever]["family"]
    elif lever in _registry.N_GPU_LEVER:
        state, reason, ev = rowpair_state(rep); fam = _registry.N_GPU_LEVER[lever]["family"]
    else:
        state, reason, ev = lever_state(lever, rep); fam = _registry.LEVERS[lever].get("family")
    if state == "off" and lever in (rep.get("levers_off") or ()):                   # dropped from the mode by MODEL_OPT_LEVERS_OFF for this run: off by that name
        reason = _registry.LEVERS_OFF
    impl, origin = _registry.IMPL[lever]
    ev.pop("impl", None)
    # strategy rides as the first positional pair: the core pins it right after origin (0.3.1+ checks it against STRATEGIES.json); the bytes are
    # `name= state= [reason=] impl= origin= strategy= arm= <evidence>` either way
    return _core_lever_line(TAG, lever, state, ("strategy", _registry.STRATEGY[lever]), ("arm", rep.get("mode")), *ev.items(), reason=reason or None, impl=impl, origin=origin), (state, reason)


def big_line(rep: dict) -> str:
    """`[af3-torch-opt] BIG base=fastest levers=graph_drop,diff_free,… disabled=none dropped=none kept=none graph_gate=stepgraph<3072tok` — the
    memory levers in force for this pred (big.selection), printed beside ACTIVE."""
    sel = rep.get("big") or {}
    from . import modes as _modes
    base = _modes.MODE_SETS.get("big")
    try:
        base_levers = _modes.kit_lever_sets().get(base, ())
    except Exception:                                       # noqa: BLE001 — the tree gate reports a missing kit; the line still prints what it knows
        base_levers = ()
    gone = _big.dropped(base_levers, sel.get("levers", ()), sel.get("settings"))       # graph levers absent from the BUILD (graph_drop gate 0 only)
    gate = (sel.get("settings") or {}).get("graph_drop_min_tokens") if "graph_drop" in sel.get("levers", ()) else None
    return f"{PREFIX} BIG " + kv(base=base, levers=sel.get("levers"), disabled=sel.get("disabled") or None,
                                  dropped=gone or None, kept=[k for v in _big.DROP.values() for k in v] if gone else None,
                                  graph_gate=(f"stepgraph<{gate}tok" if gate else None))    # graph_drop's size gate: the step graph replays below it, the step runs eager on the hoist at or above
