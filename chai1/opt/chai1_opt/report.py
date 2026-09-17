"""Observability for chai1_opt: the activation line, the ready line and the exit tally (all on stderr, prefixed ``[chai1-opt]``).

* activation — ``ACTIVE mode=<m> levels=<W1,W2,W5> eager=<tier1|none> dstep=<hoist2[,compiled,dit_attn]|none> [optin=<tf32>] levers_applied=<a,b> [not_applicable=<…>] [compile=<word>] det=<0|1>
  route=<driver|in-process> chai_lab=<v> torch=<v> gpu=<name(smNN)> [hoist=item/trunk_boundary] [big=<line>:<levers>] [n_gpu=1 sharding=none]
  [weights=pinned|unknown(k/n)] [PARTIAL off=<levers>] [notes=...] [weights_digest=fresh|cached@<utc>]`` (the last token; or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>``),
  formatted from the activation report; ``eager`` is the eager stack lever of the mode (modes.KitMode.eager); ``optin`` appears only
  when opt-in levers were requested (modes.OPTIN_LEVERS) and ``levers_applied`` then names the ones torch's state shows applied;
* numerics — one ``NUMERICS matmul=… matmul_tf32=… cudnn_tf32=… cudnn_deterministic=… cudnn_benchmark=… deterministic_algorithms=…
  warn_only=… autocast_gpu=… autocast_gpu_dtype=… autocast_cpu=…`` line per live activation: the process numerics signature the levers and
  the recipe left in force (``opt_core.precision.policy.numerics_signature``), recorded on the report as ``numerics``;
* lever lines — one ``LEVER name=<n> state=<on|off|skipped> [reason=<token>] impl=<kit file> origin=<kit|core> strategy=<id> family=<F id|LOCAL> mode=<m> route=<r>`` line
  per lever the kit composes or carries (registry.LEVERS) right after every ACTIVE / NOT ACTIVE line of a live process (never on a
  dry run): ``on`` = the kits' state shows it applied in this process, ``off`` = not in this mode's lever set and not requested (the
  reason names which: not in the mode, opt-in not requested), ``skipped`` = selected but not applied (the reason is
  mandatory: not applicable on this route, the activation's refusal, or not shown applied) — the shared core's pinned LEVER grammar
  (``opt_core.report.lever_line``: name, state, [reason], impl, origin, then ``family= mode= route=``; ``lever_lines``);
  ``notes`` carries the target-GPU mismatch note when configs/<gpu>.env's ``MODEL_OPT_TARGET_GPU`` is not the visible GPU;
* NOTE lines — right after the activation line of every route (off included): ``NOTE STACK not pinned: torch <v>+<cuda> (pinned:
  torch 2.13.0+cu130)`` when the box's torch / CUDA build is not the tree's pinned stack (stack.stack_pinning — refused by name
  instead under ``CHAI1_OPT_STRICT_STACK=1``), and ``NOTE <partial note>`` for a partial activation;
* ready — ``ready route=<r> t=<s>`` once per route when process start-up is done: imports, weights located, the recipe applied; the model
  loads lazily at the first fold (not part of ``t``);
* exit — the kits' own counters of this process, read in memory at interpreter exit: the driver's ``LOAD_LOG`` (component loads),
  ``ESM_STATS`` (W5 hits / misses), ``_MODULE_CACHE`` (W1 resident modules) and the per-input ``summary`` (n_ok / n), then the eager
  stack's ``StackHandle.stats`` — ``eager=<levers> eager_graph_fallbacks=<n> eager_events=<n>`` (+ ``dstep_ln_calls=<fast>/<all>``
  from the DSTEP add-on's own policy counters when its lever is installed), the graph-capture fallbacks
  (``graph_capture_failed:`` events of the hoisted denoiser: the stack's own counted fallback) named whenever the stack is installed,
  then ``hoist_precomputes=<n>/<items> hoist_releases=<n>`` (the item-keyed hoist, hoist.py: one precompute per fold call — per seed-fold and trunk sample —, with
  ``HOIST MISMATCH`` named on the line when the counts differ); when the lever statements never ran the tally says so (never silent).
  ``register_exit_tally()`` is called before the kit code runs.
"""
from __future__ import annotations

import atexit
import os
import sys
import time
from typing import Optional

from . import _core, hoist


def _weights_digest_token(w: dict) -> str:
    from . import stack                                                          # lazy: stack imports report
    return stack.weights_digest_token(w)

TAG = "chai1-opt"
PREFIX = f"[{_core.TAG}]"                                 # == opt_core.report.prefix(TAG)
STOCK_PREFIX = "[chai1-opt stock]"
SETTINGS_GRAMMAR = r"^\[chai1-opt(?: stock)?\] SETTINGS(?: [A-Za-z_][A-Za-z0-9_]*=\S+)+$"     # the resolved fold keywords handed to the call, one regex for both routes, once per pass
MSA_DEPTH_GRAMMAR = r"^\[chai1-opt(?: stock)?\] MSA item=\S+ chains=\d+ depth=(?:\d+(?:,\d+)*|-) dir=\S+$"   # per item: each chain's .aligned.pqt row count found in the run's MSA dir


def _tok(v) -> str:
    return "".join(str(v).split()) or "''"


def settings_line(prefix: str, kw: dict) -> str:
    """``<prefix> SETTINGS <k=v for every key of kw, sorted>`` — the resolved fold keywords actually handed to the call (stock:
    ``run_inference``'s keywords for the pass; kit: the driver's ``RUN_KW`` as written), whitespace-free tokens; once per pass, outside the
    FORWARD window."""
    return f"{prefix} SETTINGS " + " ".join(f"{k}={_tok(kw[k])}" for k in sorted(kw))


def msa_depth_line(prefix: str, item: str, chains: int, depths, msa_dir) -> str:
    """``<prefix> MSA item=<id> chains=<n> depth=<d1,d2,…|-> dir=<path|none>`` — depth_i = the row count of chain i's ``.aligned.pqt`` found in
    the run's MSA directory (0 when absent; the query row counts, so a query-only alignment reads 1)."""
    return f"{prefix} MSA item={_tok(item)} chains={int(chains)} depth={','.join(str(int(d)) for d in depths) or '-'} dir={_tok(msa_dir) if msa_dir else 'none'}"


T0 = time.time()
_TALLY = {"registered": False, "namespace": None, "route": None}


def _gpu_str(g: Optional[dict]) -> str:
    if not g or not g.get("visible"):
        return "none"
    cc = (g.get("cc") or "?").replace(".", "")
    return f"{g.get('name')}(sm{cc})"


def activation_line(rep: dict, dry_run: bool = False) -> str:
    if not rep.get("active") and not dry_run:
        if rep.get("mode") == "off" and not rep.get("reason", "").startswith(("unknown", "mode ", "opt-in")):
            return f"{PREFIX} NOT ACTIVE mode=off (stock: nothing applied)"
        return f"{PREFIX} NOT ACTIVE: {rep.get('reason')}"
    head = f"{PREFIX} DRY-RUN" if dry_run else f"{PREFIX} ACTIVE"
    fields = [f"mode={rep.get('mode')}", f"levels={rep.get('levels') or 'none'}", f"eager={rep.get('eager') or 'none'}", f"dstep={rep.get('dstep') or 'none'}"]
    if rep.get("optin"):
        fields.append(f"optin={','.join(rep['optin'])}")                     # the requested opt-in levers (modes.OPTIN_LEVERS); absent when none
    fields.append(f"levers_applied={','.join(rep.get('levers_applied') or []) or 'none'}")
    if rep.get("levers_not_applicable"):
        fields.append(f"not_applicable={','.join(rep['levers_not_applicable'])}")
    from . import modes as _modes
    cw = _modes.compile_word(rep)
    if cw:
        fields.append(f"compile={cw}")                                            # on:aoti | on:dynamo | off:user | off:no_layout_meta | off:cpu_isa_mismatch | off:card:<why> (modes.compile_word); absent for modes that never compile
    fields += [f"det={rep.get('det', 0)}", f"route={rep.get('route')}",
               f"chai_lab={rep.get('chai_lab_version')}", f"torch={rep.get('torch_version')}", f"gpu={_gpu_str(rep.get('gpu'))}"]
    if rep.get("hoist"):
        fields.append(f"hoist={rep['hoist']['keyed']}/{rep['hoist']['release']}")   # the eager denoiser's hoist key and release point (hoist.STATE)
    if rep.get("big"):
        b = rep["big"]                                                          # the memory line and the levers in force (big.py)
        fields.append(f"big={b.get('line')}:{','.join(b.get('levers') or []) or 'none'}")
        if b.get("exact"):
            fields.append(f"big_exact={b['exact']}")
    if rep.get("n_gpu") is not None:
        fields.append(ngpu_fields(rep["n_gpu"]))                                 # `n_gpu=P sharding=<none|scheme>`: the shared core's words (opt_core.mem.ngpu)
    if rep.get("weights"):
        w = rep["weights"]                                                        # the checkpoint by digest (stack.weights_match): pinned = the PINS digests; unknown = named in notes, the run proceeds
        fields.append(f"weights={w['status']}" + (f"({len(w.get('unknown') or [])}/{w.get('files')})" if w["status"] == "unknown" else ""))
    if rep.get("partial"):
        fields.append(f"PARTIAL off={','.join(rep.get('levers_off') or [])}" + (" (MODEL_OPT_LEVERS_OFF)" if rep.get("levers_off_env") else ""))
    if dry_run and rep.get("reason"):
        fields.append(f"would_refuse={rep['reason']!r}")
    if rep.get("notes"):
        fields.append("notes=" + "; ".join(rep["notes"]))
    if rep.get("weights"):
        fields.append(f"weights_digest={_weights_digest_token(rep['weights'])}")     # the last token, whitespace-free: fresh | cached@<utc> (stack.weights_digest_token)
    return head + " " + " ".join(fields)


def ngpu_fields(p) -> str:
    """The ``--n_gpu`` axis's ACTIVE / EXIT tokens (``n_gpu=1 sharding=none`` on chai1), rendered by ``opt_core.mem.ngpu`` — the kit
    types none of these words (ngpu.active_fields)."""
    from . import ngpu
    return ngpu.active_fields(int(p))


def lever_state(name: str, rep: dict):
    """``(state, reason)`` of one lever for the LEVER line: on · off (+ why it is not selected) · skipped (+ why it did not apply). Reasons
    are single tokens (the shared core's grammar splits a line on blanks): ``not_applicable_on_route:<route>`` (the driver loop's lever under
    route in-process) · ``activation_refused`` (the NOT ACTIVE line carries the refusal) · ``not_shown_applied`` (a partial activation) ·
    ``optin_not_requested`` (an implied lever the row does not carry) · ``not_in_mode:<mode>``."""
    from . import modes
    mode = rep.get("mode")
    selected = set(modes.KIT_MODES[mode].lever_names) if mode in modes.KIT_MODES else set()
    selected |= set(rep.get("optin") or [])
    if name in (rep.get("levers_applied") or []) and rep.get("active"):
        return "on", None
    b = rep.get("big") or {}
    if rep.get("active") and name in (b.get("levers") or []):
        return "on", None                                                    # a gated memory lever in force (exact / fast: applied per item above its min_n, big.py)
    if name in selected:
        if name in (rep.get("levers_not_applicable") or []):
            return "skipped", f"not_applicable_on_route:{rep.get('route')}"
        if not rep.get("active"):
            return "skipped", "activation_refused"
        return "skipped", "not_shown_applied"
    if name in modes.OPTIN_LEVERS:
        return "off", "optin_not_requested"
    return "off", f"not_in_mode:{mode}"


def lever_lines(rep: dict) -> list:
    """One LEVER line per lever of registry.LEVERS, in registry order, in the core's pinned grammar (``opt_core.report.lever_line``:
    name, state, [reason], impl, origin, strategy (``registry.STRATEGY``: the shared table's canonical id or ``LOCAL.chai1.<name>``), then
    this kit's evidence ``family= mode= route=``)."""
    from . import pairtrack, registry
    from ._core import ensure_importable
    ensure_importable()
    from opt_core.report import lever_line
    out = []
    for name, lv in registry.LEVERS.items():
        state, reason = lever_state(name, rep)
        evidence = dict(family=registry.FAMILY[name], mode=rep.get("mode"), route=rep.get("route"))
        if name in registry.PAIRTRACK_PROBES:
            evidence.update(pairtrack.evidence(name))                        # the trunk levers' counters ride their lines (served / fallback_by / ...)
        if name in registry.POSTPROC_LEVERS:                                  # the host-side levers' counters (postproc.py / featfast.py), when this process imported them
            for modname in ("chai1_opt.postproc", "chai1_opt.featfast"):
                mod = sys.modules.get(modname)
                if mod is not None:
                    evidence.update(mod.evidence(name))
        strategy = registry.STRATEGY[name]
        if name == "compiled" and state == "on":                               # the compiled step's route on this box: ahead-of-time packages present for the stack → the AOTI
            from . import modes as _modes                                      # launcher serves the crops it covers (F3.aoti_step; per crop on stderr, EXIT aoti=), else Inductor at first use;
            try:                                                               # on cc 8.0 without packages the lever is composed but steps aside by card (compile=off:card:<why>)
                cw = _modes.compile_word(rep) or ""
                if cw == "on:aoti":
                    strategy = "F3.aoti_step"
                elif cw.startswith("off:card:"):
                    state, reason = "skipped", "card:" + cw.split(":", 2)[2]
            except Exception:  # noqa: BLE001
                pass
        out.append(lever_line(TAG, name, state, reason=reason, impl=lv.kit_file, origin=registry.ORIGIN[name], strategy=strategy, **evidence))
    return out


def numerics_line(rep: dict) -> Optional[str]:
    """``[chai1-opt] NUMERICS matmul=<highest|high|medium> matmul_tf32=<b> cudnn_tf32=<b> cudnn_deterministic=<b> cudnn_benchmark=<b>
    deterministic_algorithms=<b> warn_only=<b> autocast_gpu=<b> autocast_gpu_dtype=<d> autocast_cpu=<b>`` — the process numerics signature the
    activation recorded (``rep["numerics"]``, opt_core.precision.policy.numerics_signature), or None when the report carries none."""
    sig = rep.get("numerics")
    if not sig:
        return None
    from ._core import ensure_importable
    ensure_importable()
    from opt_core.report import line, prefix
    return line(prefix(TAG), "NUMERICS", *list(sig.items()))


CORE_REFUSALS = ("reason=core_missing:", "reason=core_mismatch:", "reason=core_pin_unreadable:", "reason=producer_missing:")   # the gate's and the producers check's words (_core): the core that composes LEVER lines is what is missing


def print_activation(rep: dict, dry_run: bool = False) -> None:
    sys.stderr.write(activation_line(rep, dry_run) + "\n")
    if rep.get("stack_line"):                                                    # a torch/CUDA stack other than the pinned one is named on its own NOTE line on every route (stack.stack_pinning)
        sys.stderr.write(f"{PREFIX} NOTE {rep['stack_line']}\n")
    if rep.get("partial_note"):                                                  # a partial activation is named on its own NOTE line and proceeds (stack.partial_reason)
        sys.stderr.write(f"{PREFIX} NOTE {rep['partial_note']}\n")
    if not dry_run and not str(rep.get("reason") or "").startswith(CORE_REFUSALS):
        try:
            nl = numerics_line(rep)
            if nl:
                sys.stderr.write(nl + "\n")
            sys.stderr.write("\n".join(lever_lines(rep)) + "\n")
        except Exception as e:  # noqa: BLE001  — a lever census that cannot be composed says so on its own line, never silently absent
            sys.stderr.write(f"{PREFIX} LEVER lines failed: {e!r}\n")
    sys.stderr.flush()


def ready_line(route: str, t0: float = T0, prefix: str = PREFIX) -> str:
    return f"{prefix} ready route={route} t={time.time() - t0:.1f}"


def print_ready(route: str, t0: float = T0, prefix: str = PREFIX) -> None:
    sys.stderr.write(ready_line(route, t0, prefix) + "\n"); sys.stderr.flush()


# ------------------------------------------------------------------------------------------------------------- exit tally
def tally_fields(ns: dict) -> list:
    out = []
    lv = ns.get("levels")
    if lv is not None:
        out.append(f"levels={','.join(sorted(lv))}")
    cache = ns.get("_MODULE_CACHE")
    if isinstance(cache, dict):
        out.append(f"modules_resident={len(cache)}")
    ll = ns.get("LOAD_LOG")
    if isinstance(ll, list):
        out.append(f"loads={len(ll)}")
        secs = sum(t for _, t in ll if isinstance(t, (int, float)))
        out.append(f"load_s={secs:.1f}")
    st = ns.get("ESM_STATS")
    if isinstance(st, dict):
        out.append(f"esm_hits={st.get('hits', 0)} esm_misses={st.get('misses', 0)}")
    ec = ns.get("ESM_CACHE")
    if isinstance(ec, dict):
        out.append(f"esm_memo={len(ec)}")
    summ = ns.get("summary")
    if isinstance(summ, list):
        n_ok = sum(p.get("n_ok", 0) for p in summ if isinstance(p, dict)); n = sum(p.get("n", 0) for p in summ if isinstance(p, dict))
        out.append(f"seed_folds_ok={n_ok}/{n} inputs={len(summ)}")
    return out


def items_folded(ns: Optional[dict]) -> Optional[int]:
    """The fold calls this process made — the driver's seed-folds × their trunk samples (one run_folding_on_context call each; the
    pass summary's ``trunk_samples``, 1 when absent; the kit's ``summary``); None without one."""
    summ = (ns or {}).get("summary")
    if not isinstance(summ, list):
        return None
    return sum(p.get("n", 0) * int(p.get("trunk_samples", 1) or 1) for p in summ if isinstance(p, dict))


def items_ok(ns: Optional[dict]) -> Optional[int]:
    """The fold calls of the seed-folds this process completed (the kit's ``summary`` n_ok × trunk samples); None without a summary."""
    summ = (ns or {}).get("summary")
    if not isinstance(summ, list):
        return None
    return sum(p.get("n_ok", 0) * int(p.get("trunk_samples", 1) or 1) for p in summ if isinstance(p, dict))


def eager_tally_fields(stats: Optional[dict], levers: Optional[str], items: Optional[int] = None, ok: Optional[int] = None) -> list:
    """``eager=<levers> eager_graph_fallbacks=<n> eager_events=<n>`` from StackHandle.stats() (``diffusion_events``: (crop, what, s)
    triples; a ``graph_capture_failed:...`` triple is the hoisted denoiser's counted fallback to un-graphed steps), the DSTEP counters,
    then ``hoist_precomputes=<n>/<items> hoist_releases=<n>`` (``items``: items_folded seed-folds) + ``hoist_unreached=<k>`` when k seed-folds
    ended before their denoiser hoist (failed folds: accounting), with ``HOIST MISMATCH`` named only when a completed fold had no hoist of its
    own or one was hoisted twice (hoist.verdict)."""
    if stats is None:
        return []
    ev = stats.get("diffusion_events") or []
    fb = sum(1 for e in ev if isinstance(e, (tuple, list)) and len(e) > 1 and str(e[1]).startswith("graph_capture_failed"))
    out = [f"eager={levers or 'installed'}", f"eager_graph_fallbacks={fb}", f"eager_events={len(ev)}"]
    dc = stats.get("dstep_compile")
    if dc and dc.get("failed"):
        _w = str(dc["failed"]); _h = _w.split(":")[0].strip()
        out.append("compiled=stepped_aside:" + (":".join(x.strip() for x in _w.split(":")[:2]) if _h == "launch_error" else _h))   # the compiled step could not engage on this box: NAMED (launch_error:<ExcClass> when it raised at launch — its traceback is on stderr once), the folds ran the eager statements
    ao = stats.get("dstep_aoti")
    if ao is not None:                                                       # the compiled step's ahead-of-time packages (chai1_fastln.aoti): crops served by a package / compiled crops entered;
        out.append(f"aoti={ao.get('n_loaded', 0)}/{ao.get('n_asked', 0)}" + (f"(unusable:{len(ao.get('errors') or []) + len(ao.get('mismatch') or [])})" if (ao.get("errors") or ao.get("mismatch")) else "")
                   + (f"(realigned:{ao.get('n_realigned')})" if ao.get("n_realigned") else "")
                   + (f"(refused:{','.join(sorted({r.split(':', 1)[1].split(':')[0] for r in ao['refused']}))}:{len(ao['refused'])})" if ao.get("refused") else ""))   # refused by name (no_layout_meta | layout_unproducible | cpu_isa_mismatch): the card's no-package route served   # the rest compiled through Dynamo; realigned: inputs copied to aligned buffers (named on stderr)
        if ao.get("n_aside") and ao.get("aside_reason") and not (dc and dc.get("failed")):   # … or, on cc 8.0, stepped aside per crop by card: named once like a compile that could not engage
            out.append("compiled=stepped_aside:" + str(ao["aside_reason"]))
    da = stats.get("dstep_dit_attn")
    if da:
        if da.get("stepped_aside"):
            out.append("dit_attn=stepped_aside:" + str(da["stepped_aside"]).split(":")[0].split(" ")[0])   # dit_exact cannot serve on this box, by NAME (e.g. no_dit_exact_row_cc80): the calls kept their statement
        elif da.get("served_by"):
            out.append(f"dit_attn={da['served_by']}:{da.get('served', 0)}/{da.get('calls', 0)}" + (f"(refused:{da['refusal']})" if da.get("refusal") else ""))   # row served : DiT calls served / all SDPA calls of the steps
        elif da.get("refusal") or da.get("errors"):
            out.append(f"dit_attn=not_served:{da.get('refusal') or da.get('errors')}")
    ln = stats.get("dstep_ln")
    if ln is not None:                                                       # the DSTEP add-on's own policy counters (LNPolicy n_fast, n_slow)
        out.append(f"dstep_ln_calls={ln.get('n_fast', 0)}/{ln.get('n_fast', 0) + ln.get('n_slow', 0)}")
    hs = stats.get("hoist")
    if hs is not None:                                                       # the item-keyed hoist's counters (hoist.stats): one precompute per item that reached its denoiser
        out.append(f"hoist_precomputes={hs['n_precompute']}/{'?' if items is None else items} hoist_releases={hs['n_release']}")
        k = hoist.unreached(hs, items, ok)
        if k:
            out.append(f"hoist_unreached={k}")                                # seed-folds that ended (failed) before their denoiser hoist: accounting, not a stale hoist
        why = hoist.verdict(hs, items, ok)
        if why:
            out.append(f"HOIST MISMATCH {why}")
    if "error" in stats:
        out.append(f"eager_stats_error={stats['error']!r}")
    return out


def optin_tally_fields() -> list:
    """The levers' own exit counters: the trunk levers' per-call counters (+ ``PAIRTRACK_GATE_REFUSED`` when their gate refuses), the
    host-side levers' (+ ``POSTPROC_GATE_REFUSED``), big's chunk-site census, ``alloc_effective=``, the ESM memo scope and the forward
    timer's totals — empty for levers that never applied in this process."""
    from . import alloc, pairtrack
    out = []
    out += pairtrack.tally_fields()                                          # the trunk levers' per-call counters (empty when none is installed)
    for modname in ("chai1_opt.postproc", "chai1_opt.featfast"):               # the host-side levers' counters + their gate word — read only if this process imported them
        mod = sys.modules.get(modname)                                           # (the tally runs at interpreter exit: no first import there)
        if mod is None:
            continue
        out += mod.tally_fields()
        why = mod.verdict()
        if why:
            out.append("POSTPROC_GATE_REFUSED=" + why.replace(" ", ";"))
    why = pairtrack.verdict()
    if why:
        out.append("PAIRTRACK_GATE_REFUSED=" + why.replace(" ", ";"))
    from . import forward_timer, stack as _stack
    try:                                                                        # big's chunk-site census: msa_chunk= opm_chunk= trunk_chunk= (<served>/<items> | skipped:<word>)
        from . import big as _big
        out = out + _big.chunk_census_fields()
    except Exception as e:  # noqa: BLE001 — named on the line, never fatal
        out.append(f"chunk_census_error={type(e).__name__}")
    return out + alloc.tally_fields() + _stack.esm_scope_tally_fields() + forward_timer.tally_fields()   # … esm_memo_scope=<s> esm_memo_clears=<n> forward_calls=<n> forward_s_total=<sec> last


def exit_tally_line(pid: Optional[int] = None) -> str:
    pid = os.getpid() if pid is None else pid
    ns = _TALLY["namespace"]
    stats = levers = None; tally_error = None
    try:
        from . import stack
        if ns is None:
            ns = stack.kit_namespace()
        stats = stack.eager_stats()
        levers = (stack._STATE["report"] or {}).get("eager") if stats is not None else None
    except Exception as e:  # noqa: BLE001  — named on the line: a tally that could not read the kits' state says so
        tally_error = repr(e)
    route = _TALLY["route"] or "in-process"
    axis = []
    try:
        p = (stack._STATE["report"] or {}).get("n_gpu")
        if p is not None:
            axis = [ngpu_fields(p)]                                              # the resource axis the run folded on: `n_gpu=P sharding=…`
    except Exception as e:  # noqa: BLE001
        tally_error = (tally_error + "; " if tally_error else "") + f"n_gpu:{e!r}"
    if not ns:
        return (f"{PREFIX} EXIT pid={pid} route={route} " + " ".join(axis + ["no lever counters: the kit's lever statements never ran in this process"])
                + (f" tally_error={tally_error}" if tally_error else ""))
    msa = []
    try:
        f_ = stack.msa_form_fields()                                                 # the driver's MSA-form census (`msa_form_directory=<n> msa_form_none=<n>`)
        if f_:
            msa = [f_]
    except Exception as e:  # noqa: BLE001
        tally_error = (tally_error + "; " if tally_error else "") + f"msa_form:{e!r}"
    return (f"{PREFIX} EXIT pid={pid} route={route} source=memory " + " ".join(axis + tally_fields(ns) + eager_tally_fields(stats, levers, items_folded(ns), items_ok(ns)) + optin_tally_fields() + msa)
            + (f" tally_error={tally_error}" if tally_error else ""))


def _print_exit_tally() -> None:
    try:
        line = exit_tally_line()
    except Exception as e:  # noqa: BLE001
        line = f"{PREFIX} EXIT tally failed: {e!r}"
    try:
        sys.stderr.write(line + "\n"); sys.stderr.flush()
    except Exception:  # noqa: BLE001  — best effort at interpreter exit: stderr may already be closed; nothing else can be told
        pass


def register_exit_tally(namespace: Optional[dict] = None, route: Optional[str] = None) -> None:
    """Print the exit tally at interpreter exit (once per process). ``namespace``: the dict the kit's statements run in."""
    if namespace is not None:
        _TALLY["namespace"] = namespace
    if route is not None:
        _TALLY["route"] = route
    if _TALLY["registered"]:
        return
    _TALLY["registered"] = True
    atexit.register(_print_exit_tally)
