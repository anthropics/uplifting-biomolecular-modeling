"""Observability for rosettafold3_opt: the activation line, the applied line and the exit tally (all on stderr, prefixed
``[rosettafold3-opt]``).

* activation — ``ACTIVE mode=<m> row=<switches> tree=<state(n/5)> foundry=<v>@<commit8> torch=<v> gpu=<name(smNN)>
  levers=<a,b,...> applied=deferred`` (or ``DRY-RUN ...`` / ``NOT ACTIVE: <reason>``), from the report ``enable()`` / ``status()`` return;
* lever — at exit, one line per registry lever in the core's pinned grammar (``opt_core.report.lever_line``; ``lever_lines``):
  ``LEVER name=<house name> state=<on|off|skipped> [reason=<token>] impl=<file|kernel> origin=<kit|core> strategy=<F<k>.x|LOCAL.x>
  <evidence k=v>`` — ``on`` only when the lever's own probe (registry.py) shows it acted in THIS process; the same states go into the
  tally JSON as ``levers``;
* applied — ``APPLIED rf3.graph_flags RF3_CUDAGRAPH=1 RF3_GRAPH_SAFE_OPS=True RF3_HOIST=True ...`` once, when the kit's own
  ``describe()`` has been read after the kit file executed (stack.applied_check);
* FPF applied — ``FPF APPLIED arm=<arm> trimul=<mode> triattn=<..> transition=<..> apb=<..> trunk_graph=<b> dattn=<b> res=<b>
  levers=<...>`` once, when the add-on's adapter has applied the row's arm (stack.fpf_apply), from its own ``describe_v2()``;
* exit — ``EXIT mode=<m> rollouts=<n> entries_last=<k> capture=<...> describe=<...>``: the kit module's own counters
  (``HOIST_STATS``, ``LAST_CAPTURE``, ``describe()``) read in memory at interpreter exit; when the
  kit module was never imported the tally says so (never silent). Under an FPF arm the line continues ``fpf=<arm> served=<n>
  fallback=<...> errors=<n> tg_replays=<n> tg_fallbacks=<...> kernel_errors=<...> levers_acted=<...> levers_silent=<...> ok=<b>
  [levers_budget_skipped=fpf_tg(max_i=<n>,skipped:<I>=<n>)] [levers_unreached=<lever>(no_call_reached:<zero_items|early_stop>)[;…]] [levers_size_gated=fpf_trimul(N<101=<n>,…)[;fpf_ttr(M<4096=<n>)][;fpf_res(unfused=<n>)]]
  [upstream=items:<n>,early_stopped:<n>,rolled_out:<n>[,no_rollout:<zero_items|early_stop>]]``:
  the adapter's own counters (``describe()``, ``describe_v2()``: the trimul kernel's served / fallback / error counts, the trunk graph's
  replays and its own fallbacks — capture_failed, shape_gave_up, grad_or_off, evictions — and every ``kernel-error`` key of the
  component counters) and, per FPF lever of the row, its registry probe — a lever that never acted is named (``levers_silent``); a
  lever every call of which took a named route away from it by the input's size is named apart and is not a failure (``levers_budget_skipped``:
  the trunk graph above its token budget; ``levers_size_gated``: a kernel below its designed size gate — size_gated_levers); an
  error count, a gated trunk-graph fallback (``TG_FALLBACKS_GATED``), a kernel error or a silent lever makes ``ok`` false, and
  ``pred`` fails the run on it (fold.py): a fast run whose kernels fell back to stock is a named failure, never a quiet stock run. ``register_exit_tally()`` also writes the tally
  as JSON to the path in ``ROSETTAFOLD3_OPT_TALLY_FILE`` when set (``pred`` reads it for its verdict).
"""
from __future__ import annotations

import atexit
import re
import json
import os
import sys

from . import _core

_core_report = _core.load("report")

TAG = "rosettafold3-opt"
PREFIX = "[rosettafold3-opt]"
PREFIX_STOCK = "[rosettafold3-opt stock]"
TG_FALLBACKS_GATED = ("capture_failed", "shape_gave_up", "grad_or_off")   # the adapter's trunk-graph degradations (fpf_rf3_adapter.py _recycler_forward_tg) that close the gate;

# A kernel lever's DESIGNED size gate is part of its contract: a fold whose every call of the lever was declined by that gate (and by nothing
# else) names the lever `levers_size_gated` on the EXIT line and `LEVER … state=skipped reason=below_gate` — reached and routed by the rule,
# not silent (fpf_tally). The decline words are the kernels' / the adapter's own census keys:
TRIMUL_TOKEN_FLOOR = re.compile(r"N<\d+")                                       # fpf_trimul_v4 below its token floor (generic / trimul: N_MIN = 101)
TRIMUL_TRACK_CLASS = re.compile(r"c=\d+/d=\d+")                                 # the pair track the trimul kernel does not serve (the c=64 template track: on every input, never alone a size gate)
TTR_ROW_FLOOR = re.compile(r"fallback:(?:M<\d+|[NI]<=?\d+\S*)")                                    # the fused transition below its size floor (the provider's cells: pair transitions of small inputs run the stock statements by name)
TTR_CHANNEL_CLASS = re.compile(r"fallback:C=\d+,HID=\d+")                          # a transition class the kernel does not serve (the single track's C=384 transitions: on every input, never alone a size gate)


def ttr_size_decline(reason: str) -> bool:
    """A transition decline that is a property of the input or the model, not of the machine: below the row floor, or an unserved channel class."""
    return bool(TTR_ROW_FLOOR.fullmatch(str(reason)) or TTR_CHANNEL_CLASS.fullmatch(str(reason)))


def trimul_size_decline(reason: str) -> bool:
    """A trimul decline that is a property of the input or the model, not of the machine: below the token floor, or the unserved pair track."""
    return bool(TRIMUL_TOKEN_FLOOR.fullmatch(str(reason)) or TRIMUL_TRACK_CLASS.fullmatch(str(reason)))


def size_gated_levers(silent: list, desc: dict, desc2: dict, trunk_sizes=None) -> dict:
    """``{lever: {word: calls}}`` for the silent FPF levers whose silence is their size gate: ``fpf_trimul`` — served 0, no adapter error, at
    least one ``N<…`` decline and every decline a size word (the floor, or the unserved c=64 track that every input has); ``fpf_ttr`` — no
    ``served:`` transition call, at least one ``fallback:M<…`` decline and every decline a size word (the row floor, or an unserved channel
    class — the single track's C=384 transitions that every input has: ttr_size_decline); ``fpf_res`` — the residual fusion rides those two
    kernels' epilogues at the pair block's sites: no fused call, unfused calls counted (the block forward was reached), the trimul size-gated,
    and the transition declines likewise all size words with the row floor among them (a transition served elsewhere — the MSA track's rows —
    does not fuse a pair-block residual, so its served count does not speak here)."""
    out: dict = {}
    fb = {str(k): int(v) for k, v in (desc.get("fallback") or {}).items()}
    if ("fpf_trimul" in silent and not int(desc.get("served") or 0) and not int(desc.get("errors") or 0) and fb
            and any(TRIMUL_TOKEN_FLOOR.fullmatch(k) for k in fb) and all(trimul_size_decline(k) for k in fb)):
        out["fpf_trimul"] = dict(fb)
    tr = {str(k): int(v) for k, v in ((desc2.get("counts") or {}).get("transition") or {}).items()}
    tr_fb = {k: v for k, v in tr.items() if k.startswith("fallback:")}
    tr_by_size = bool(tr_fb) and any(TTR_ROW_FLOOR.fullmatch(k) for k in tr_fb) and all(ttr_size_decline(k) for k in tr_fb)   # the floor among them, nothing but size words
    if "fpf_ttr" in silent and not any(k.startswith("served:") for k in tr) and tr_by_size:
        out["fpf_ttr"] = {k[len("fallback:"):]: v for k, v in tr_fb.items()}
    res = desc2.get("res") or {}
    if ("fpf_res" in silent and not int(res.get("fused") or 0) and int(res.get("unfused") or 0) > 0 and "fpf_trimul" in out and tr_by_size):
        out["fpf_res"] = {"unfused": int(res.get("unfused") or 0)}
    sizes = sorted(int(n) for n in (trunk_sizes or ()))
    for lever, sec in (("fpf_xatt", "xatt"), ("fpf_xmul", "xmul")):                              # the provider-row components: every call of the input went to
        words = census_size_words(desc2.get(sec) or {})                                          # the stock op BY NAME below the library's own threshold
        if lever in silent and words:                                                            # (census passthrough:S<=<t> / passthrough:N<=<t>) — reached and
            out[lever] = words                                                                   # routed by the named size rule, not silent
        elif (lever in silent and not census_calls(desc2.get(sec) or {}) and sizes               # … or never reached at all: every trunk of the process was at or
              and sizes[-1] <= LIBRARY_SMALL_N):                                                 # below the library's own small-size threshold, where the stock op
            out[lever] = {"unreached:I<=%d" % LIBRARY_SMALL_N: sizes[-1]}                        # takes its reference path before the kit's hook (named, by size)
    ex = desc2.get("smsa") or (desc2.get("msa") or {}).get("exact") or {}
    ex_fb = {str(k): int(v) for k, v in (ex.get("fallback") or {}).items()}
    if "fpf_smsa" in silent and not int(ex.get("served") or 0) and ex_fb and all(SIZE_ROUTE_WORD.fullmatch(k) for k in ex_fb):
        out["fpf_smsa"] = ex_fb                                                                  # the exact PWA cell below its size floor (floor:S<..._or_rows<...)
    msa = desc2.get("msa") or {}
    units = (msa.get("counts") or {}) if isinstance(msa.get("counts"), dict) else {}
    unit_fb = {u: {str(k): int(v) for k, v in ((c or {}).get("fallback") or {}).items()} for u, c in units.items()}
    if ("fpf_msa" in silent and not int(msa.get("served") or 0) and unit_fb and all(unit_fb.values())
            and all(SIZE_ROUTE_WORD.fullmatch(k) for fb_u in unit_fb.values() for k in fb_u)):
        out["fpf_msa"] = unit_fb                                                                 # every MSA unit below its token floor (I<...) or left at stock by its card row
    return out


SIZE_ROUTE_WORD = re.compile(r"passthrough:[SNI]<=?\d+|floor:\S+|[SNI]<=?\d+\S*|rows<\d+\S*|cc\d+\.\d+:\S+")   # the named size / card routes a component counts


def trunk_sizes(budget_census: dict, desc2: dict) -> list:
    """The token counts of every trunk this process ran: the budget rule's per-size census (``by_key``: ``graphed:<I>`` / ``skipped:<I>``, tgbudget.py)
    and the trunk graph's own capture records (``trunk_graph.captures[].I``) — whichever the arm has."""
    sizes = set()
    for k in ((budget_census or {}).get("by_key") or {}):
        kind, _, n = str(k).partition(":")
        if kind in ("graphed", "skipped") and n.isdigit():
            sizes.add(int(n))
    for cap in ((desc2.get("trunk_graph") or {}).get("captures") or []):
        try:
            sizes.add(int(cap.get("I")))
        except Exception:
            pass
    return sorted(sizes)


LIBRARY_SMALL_N = 100          # cuEquivariance's own fallback threshold (CUEQ_TRIATTN_FALLBACK_THRESHOLD / CUEQ_TRIMUL_FALLBACK_THRESHOLD default): at or below it
                               # the stock op runs its torch reference path and the provider-row hooks (xatt / xmul) are not reached


def census_calls(sec: dict) -> dict:
    """``{word: calls}`` of a provider-row component's describe_v2 section (census words -> counts; ``on`` / ``served`` / cfg keys excluded)."""
    return {str(k): int(v) for k, v in sec.items() if k not in ("on", "served", "word", "cfg") and isinstance(v, int) and not isinstance(v, bool) and v}


def census_size_words(sec: dict) -> dict:
    """``{word: calls}`` when a provider-row component's census holds calls and every one of them is a named size route (nothing served, no
    refusal, no error); ``{}`` otherwise. ``sec`` is the component's describe_v2 section: census words -> counts plus ``on`` / ``served``."""
    calls = census_calls(sec)
    if not calls or int(sec.get("served") or 0):
        return {}
    return dict(calls) if all(SIZE_ROUTE_WORD.fullmatch(k) for k in calls) else {}
                                                                          # "evictions" (the graph pool's LRU, a multi-shape process) is printed, not gated


def _join(items) -> str:
    return ",".join(str(x) for x in items) if items else "none"


gpu_label = _core_report.gpu_label                    # ``name(smNN)`` | ``none``: the core's one label of a GPU probe dict


def _versions(rep: dict) -> str:
    up = rep.get("upstream") or {}
    c = (up.get("commit") or "")[:8] or "?"
    return f"foundry={up.get('foundry')}@{c} torch={up.get('torch')}"


def active_line(rep: dict) -> str:
    head = "DRY-RUN" if rep.get("dry_run") else ("ACTIVE" if rep.get("active") else "NOT ACTIVE")
    if not rep.get("active") and not rep.get("dry_run"):
        return f"{PREFIX} NOT ACTIVE: {rep.get('reason')}"
    extras = ""
    if rep.get("withheld"):                               # MODEL_OPT_LEVERS_OFF (leversoff.py): the names the request removed from the mode, request order
        from . import leversoff as _leversoff
        extras += _leversoff.token(rep["withheld"])
    if rep.get("absent"):                                 # MODEL_OPT_LEVERS_OFF words with no lever class in this kit (compile): accepted, ` compile=none`
        from . import leversoff as _leversoff
        extras += _leversoff.absent_token(rep["absent"])
    tree = rep.get("tree_line") or "tree=?"
    tree = tree.split(" ")[0]                            # `tree=<state>(n/5)`: the site-packages path stays on the TREE line
    fpf = rep.get("fpf") or {}
    fpf_s = f" fpf={fpf.get('arm')}" if fpf else ""
    return (f"{PREFIX} {head} mode={rep.get('mode')} row={rep.get('switch_line')}{fpf_s} {tree} {_versions(rep)} gpu={gpu_label(rep.get('gpu'))} "
            f"levers={_join(rep.get('levers'))} applied={rep.get('applied')}{extras}"
            + (f" reason={rep.get('reason')}" if rep.get("dry_run") and rep.get("reason") else "")
            + f" {ngpu_fields(rep)}")                                                          # the resource axis closes the line (n_gpu=P sharding=<scheme|none>)


def ngpu_fields(rep_or_tally) -> str:
    """``n_gpu=P sharding=<scheme|none>`` — the memory mode's resource-axis tokens, worded by ``opt_core.mem.ngpu.active_fields`` (the one
    producer; P=1: ``n_gpu=1 sharding=none``). Carried by the ACTIVE line and the EXIT line of every kit-row process."""
    ngpu = _core.load("mem.ngpu")
    p = int((rep_or_tally or {}).get("n_gpu") or 1)
    return ngpu.active_fields(p, (rep_or_tally or {}).get("sharding") if p > 1 else "rowpair") if p > 1 else ngpu.active_fields(1)


def print_active(rep: dict) -> None:
    print(active_line(rep), file=sys.stderr, flush=True)


def print_not_active(rep: dict) -> None:
    print(f"{PREFIX} NOT ACTIVE: {rep.get('reason')}", file=sys.stderr, flush=True)


def applied_line(rep: dict) -> str:
    d = rep.get("describe") or {}
    return f"{PREFIX} APPLIED rf3.graph_flags " + " ".join(f"{k}={v}" for k, v in d.items()) + f" levers={_join(rep.get('levers_applied'))}"


def print_applied(rep: dict) -> None:
    print(applied_line(rep), file=sys.stderr, flush=True)


def fpf_applied_line(rep: dict) -> str:
    f = rep.get("fpf") or {}
    cfg = f.get("cfg") or {}
    return (f"{PREFIX} FPF APPLIED arm={f.get('arm')} " + " ".join(f"{k}={v}" for k, v in cfg.items())
            + f" fpf_levers={_join([lv for lv in (rep.get('levers_applied') or []) if str(lv).startswith('fpf_')])}")


def print_fpf_applied(rep: dict) -> None:
    print(fpf_applied_line(rep), file=sys.stderr, flush=True)


def xtr_line(rep: dict) -> str:
    """The xtr lever at its trigger: ``[rosettafold3-opt] XTR on=… construction=… serve=<cells> [floor=rows<<n>>:route:prev] routes=<key:route,…>
    cells_sha256=… prev=<the forward it wrapped>`` (``floor`` only on a device with a row floor, pf.CARD_ROWS)."""
    x = rep.get("xtr") or {}
    routes = ",".join(f"{k}:{v}" for k, v in sorted((x.get("routes") or {}).items()))
    floor = f" floor=rows<{x['floor']}:route:prev" if x.get("floor") is not None else ""      # a device with a row floor (pf.CARD_ROWS) names it; none: no field
    return (f"{PREFIX} XTR on={x.get('on')} construction={x.get('construction')} serve={_join(x.get('serve'))}{floor} routes={routes} "
            f"cells_sha256={(x.get('cells_sha256') or '')[:12]} prev={x.get('prev')} executes_from={x.get('impl')}"
            + ("" if x.get("on") else f" reason={str(x.get('reason')).replace(' ', '_')}"))


def print_xtr(rep: dict) -> None:
    print(xtr_line(rep), file=sys.stderr, flush=True)


def mkdit_line(rep: dict) -> str:
    """The mkdit lever at the arm's apply: ``[rosettafold3-opt] MKDIT on=… executes_from=<the carried mk2.py> files=<name:sha12,…> min_tokens=… config=<tiles> gpu=<name/cc>``."""
    m = rep.get("mkdit") or {}
    files = ",".join(f"{k.rsplit('/', 1)[-1]}:{v[:12]}" for k, v in sorted((m.get("files_sha256") or {}).items()))
    cfg = ",".join(f"{k}={v}" for k, v in (m.get("config") or {}).items())
    g = m.get("gpu") or {}
    return (f"{PREFIX} MKDIT on={m.get('on')}"
            f" executes_from={m.get('impl')} files={files} min_tokens={m.get('min_tokens')} config={cfg} "
            f"gpu={str(g.get('name')).replace(' ', '_')}/cc{g.get('cc')}"
            + ("" if m.get("on") else f" reason={str(m.get('reason')).replace(' ', '_')}"))


def print_mkdit(rep: dict) -> None:
    print(mkdit_line(rep), file=sys.stderr, flush=True)


def kernels_line(rep: dict) -> str:
    """The routed kernels: each name and the core copy it resolves to (the route held before any import), the run-time import decision."""
    ks = (rep.get("fpf") or {}).get("kernels") or {}
    parts = [f"{k}={v.get('resolved')}" for k, v in ks.items()]
    ri = {n: w for k, v in ks.items() for n, w in (v.get("runtime_imports") or {}).items()}
    return f"{PREFIX} KERNELS routed=" + ",".join(ks) + " " + " ".join(parts) + (" runtime_imports=" + ",".join(f"{n}:{w}" for n, w in ri.items()) if ri else "")


def print_kernels(rep: dict) -> None:
    print(kernels_line(rep), file=sys.stderr, flush=True)


def big_line(rep: dict) -> str:
    """The big mode's line at activation (the levers expected; the APPLIED line follows at the trigger): ``[rosettafold3-opt] BIG base=… fpf_arm=… disengaged=… expected=… off=… on=… allocator=… trigger=…``."""
    b = rep.get("big") or {}
    return (f"{PREFIX} BIG base={b.get('base')} fpf_arm={b.get('fpf_arm')} disengaged={_join(sorted(b.get('disengaged') or []))} expected={_join(b.get('expected'))} "
            f"off={_join(b.get('off_by_flag'))} on={_join(b.get('on_by_flag'))} allocator={b.get('allocator_writer')} trigger={b.get('trigger')} applied={b.get('applied')}")


def print_big(rep: dict) -> None:
    print(big_line(rep), file=sys.stderr, flush=True)


def print_mem(rep: dict) -> None:
    """The memory policy line (mem.py), printed after ACTIVE (and KERNELS) by every kit-row activation."""
    from . import mem as _mem
    print(_mem.line((rep.get("mem") or {}).get("policy"), (rep.get("mem") or {}).get("fpf_tg_max")), file=sys.stderr, flush=True)


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)


def fpf_probe(name: str, probe: tuple, desc: dict, desc2: dict) -> int:
    """One registry probe of an FPF lever against the adapter's own records (registry.py docstring): the count it names."""
    kind = probe[0]
    if kind == "fpf_served":
        return int(desc.get("served") or 0)
    if kind == "fpf_count":
        counts = (desc2.get("counts") or {}).get(probe[1]) or {}
        return int(sum(int(v) for k, v in counts.items() if str(k).startswith(probe[2])))
    if kind == "fpf_v2":
        return int(((desc2.get(probe[1]) or {}).get(probe[2])) or 0)
    if kind == "fpf_v2_sum":
        sec = desc2.get(probe[1]) or {}
        return int(sum(int(sec.get(k) or 0) for k in probe[2]))
    raise ValueError(f"lever {name}: probe {probe!r} is not an FPF probe")


def fpf_tally(rep: dict | None) -> dict:
    """The adapter's own records of this process (never imports it: reads sys.modules) and, per FPF lever of the row, whether it acted."""
    f = (rep or {}).get("fpf") or {}
    adp = sys.modules.get("fpf_rf3_adapter")
    out = {"arm": f.get("arm"), "applied": f.get("applied"), "adapter_imported": adp is not None, "reason": f.get("reason")}
    if adp is None:
        out["ok"] = False
        out["reason"] = out["reason"] or "fpf_rf3_adapter never imported in this process (the arm was not applied: no fold ran here)"
        return out
    try:
        desc, desc2 = dict(adp.describe()), dict(adp.describe_v2())
    except Exception as e:                                   # pragma: no cover
        out.update({"ok": False, "reason": f"adapter describe failed: {e!r}"})
        return out
    out["describe"] = _plain(desc)
    out["describe_v2"] = _plain(desc2)
    from . import tgbudget as _tgb
    if _tgb.STATE["on"]:                                                                     # the trunk-graph token budget (tgbudget.py): graphed / skipped by size, a named route
        out["tg_budget"] = _plain(_tgb.describe())
    from .registry import LEVERS
    acted, silent, budget_skipped = [], [], []
    withheld = set((rep or {}).get("withheld") or [])
    for lv in (rep or {}).get("levers") or []:
        if not str(lv).startswith("fpf_") or lv in withheld:                                       # a lever the run withheld by name (MODEL_OPT_LEVERS_OFF) is off by design, never silent
            continue
        n = fpf_probe(lv, LEVERS[lv].probe, desc, desc2)
        (acted if n > 0 else silent).append(lv)
    bc = (out.get("tg_budget") or {}).get("census") or {}
    if "fpf_tg" in silent and int(bc.get("skipped") or 0) > 0 and int(bc.get("graphed") or 0) == 0:   # every trunk call of this process was above the token budget:
        silent.remove("fpf_tg"); budget_skipped.append("fpf_tg")                                     # the graph was reached and routed by the named rule (tgbudget.py), not silent;
        out["tg_budget_skipped"] = {"max_i": (out.get("tg_budget") or {}).get("max_i"),              # a graph that was handed calls (graphed > 0) and never replayed stays silent
                                    "by_key": _plain(bc.get("by_key") or {})}
    gates = size_gated_levers(silent, desc, desc2, trunk_sizes=trunk_sizes(bc, desc2))               # a lever whose every call its own size gate declined:
    size_gated = [n for n in silent if n in gates]                                                   # reached and routed by the named rule, not silent
    silent = [n for n in silent if n not in gates]
    if gates:
        out["size_gated"] = {n: gates[n] for n in size_gated}
    from . import upstream as _upstream                                                              # a lever whose seam UPSTREAM never reached in this process — nothing to predict
    why = _upstream.no_rollout_reason() if silent else None                                          # (skip_existing) or every input early-stopped before its roll-out — is named
    unreached = list(silent) if why else []                                                          # `no_call_reached:<zero_items|early_stop>`, not silent; a process where an
    silent = [] if why else silent                                                                   # input DID roll out keeps the fail-closed rule
    if unreached:
        out["unreached"] = {n: f"no_call_reached:{why}" for n in unreached}
    tg = desc2.get("trunk_graph") or {}
    counts = desc2.get("counts") or {}
    kernel_errors = {f"{kind}:{k}": int(v) for kind, d in counts.items() if isinstance(d, dict) for k, v in d.items() if str(k).startswith("kernel-error")}
    out.update({"served": int(desc.get("served") or 0), "fallback": _plain(desc.get("fallback") or {}), "errors": int(desc.get("errors") or 0),
                "tg_replays": int(tg.get("replays") or 0), "tg_fallbacks": _plain(tg.get("fallbacks") or {}), "kernel_errors": kernel_errors,
                "levers_acted": acted, "levers_silent": silent, "levers_budget_skipped": budget_skipped, "levers_size_gated": size_gated,
                "levers_unreached": unreached})
    reasons = []
    if f.get("applied") != "configured":
        reasons.append(f"arm not applied ({f.get('applied')})")
    if out["errors"]:
        reasons.append(f"{out['errors']} adapter error(s)")
    degraded = {k: v for k, v in out["tg_fallbacks"].items() if k in TG_FALLBACKS_GATED and v}
    if degraded:
        reasons.append("trunk-graph fallback(s): " + ",".join(f"{k}={v}" for k, v in degraded.items()))
    if kernel_errors:
        reasons.append("kernel error(s): " + ",".join(f"{k}={v}" for k, v in kernel_errors.items()))
    if silent and (rep or {}).get("mode") == "big":
        out["disengaged_by_property"] = list(silent)                  # on the big row a fast kernel that served no call because every call took a named route away from it (the FPF trimul's counted core declines — token floor, no cell row for the card, a dtype the cell does not serve (neither bf16 nor fp32); the fused residual adds that ride the trimul / transition kernels; every arm kernel under --n_gpu > 1, where the row-sharded pair stack owns the sites) is a lever PROPERTY the census names, never a failure
    elif silent:
        reasons.append("silent lever(s): " + ",".join(silent))
    out["ok"] = not reasons
    out["reason"] = "; ".join(reasons) if reasons else None
    return out


def tally(rep: dict | None) -> dict:
    """The kit module's own records of this process (never imports it: reads sys.modules); the adapter's under an FPF arm."""
    gf = sys.modules.get("rf3.graph_flags")
    _fy = sys.modules.get("foundry")                                                                                # the cuEquivariance flag of THIS process
    out = {"mode": (rep or {}).get("mode"), "row": (rep or {}).get("row"), "graph_flags_imported": gf is not None, "withheld": list((rep or {}).get("withheld") or []),
           "absent": list((rep or {}).get("absent") or []),
           "cueq": (bool(getattr(_fy, "SHOULD_USE_CUEQUIVARIANCE", False)) if _fy is not None else "na"),
           "n_gpu": int((rep or {}).get("n_gpu") or 1), "sharding": (rep or {}).get("sharding") or "none"}
    if (rep or {}).get("rowpair") is not None:
        out["rowpair"] = dict(rep["rowpair"])                                                                           # the row-sharding adapter's install record (JSON only)
        rp = sys.modules.get(__name__.rsplit(".", 1)[0] + ".rowpair")
        if rp is not None and hasattr(rp, "state"):
            out["rowpair"]["state"] = rp.state()
    if (rep or {}).get("template") is not None:
        out["template"] = dict(rep["template"])                                                                         # the template census (templ.py; JSON only)
    if (rep or {}).get("fpf"):
        out["fpf"] = fpf_tally(rep)
        out["kernels"] = {"routed": {k: v.get("resolved") for k, v in (rep["fpf"].get("kernels") or {}).items()},          # the route held at activation
                          "executed_from": rep["fpf"].get("kernels_imported")}                                                # the census after apply (JSON only)
    from . import mem as _mem
    if _mem.report() is not None:
        out["mem"] = _mem.report()                                                                                       # the release counters (JSON only)
    from . import big as _big
    if _big.state() is not None:
        out["big"] = _big.state()                                                                                 # the big record + census + exit verdict (JSON only)
    if "dtk" in ((rep or {}).get("levers") or []):
        from . import dtk as _dtk
        out["dtk"] = _dtk.describe()                                                                                    # the dtk lever: gate census + ok/reason
    if "xtr" in ((rep or {}).get("levers") or []):
        from . import pf as _pf
        out["xtr"] = _pf.describe()
    if "mkdit" in ((rep or {}).get("levers") or []):
        from . import mkdit as _mkdit
        out["mkdit"] = _mkdit.describe()
    if "hostlean" in ((rep or {}).get("levers") or []):
        from . import hostlean as _hostlean
        out["hostlean"] = _hostlean.describe()
    if "prefetch" in ((rep or {}).get("levers") or []):
        from . import prefetch as _prefetch
        out["prefetch"] = _prefetch.describe()                                                                          # the persistent featurizer: items served by the helper / in-process by name, first-input verdict, shadow
    if "awrite" in ((rep or {}).get("levers") or []):
        from . import awrite as _awrite
        out["awrite"] = _awrite.describe()                                                                              # the asynchronous writer: files written by the side process / in-process by name, errors, flush wait                                                                          # validation_step's symmetry resolutions: skipped / kept-by-name census
    if gf is not None and hasattr(gf, "levers_state"):
        out["graph_levers"] = dict(gf.levers_state())                                                                   # the roll-out's runtime levers (warm) + capture / warm-up counts
    if "confhoist" in ((rep or {}).get("levers") or []):
        from . import confhoist as _confhoist
        out["confhoist"] = _confhoist.describe()                                                                        # the confidence-head prologue hoist: census + refusal
    if "confln" in ((rep or {}).get("levers") or []):
        from . import confhoist as _confhoist
        out["confln"] = _confhoist.describe_ln()                                                                        # the prologue's whole-tensor layer norms by var_mean (fast class): calls + refusal                                                                                # the mkdit lever: files, gate census + ok/reason
    if gf is not None:
        try:
            out["describe"] = dict(gf.describe())
        except Exception as e:                               # pragma: no cover
            out["describe_error"] = repr(e)
        for name in ("HOIST_STATS", "LAST_CAPTURE"):
            v = getattr(gf, name, None)
            if isinstance(v, dict):
                out[name] = {k: (val if isinstance(val, (int, float, str, bool, type(None))) else str(val)) for k, val in v.items()}
    from . import upstream as _upstream
    out["upstream"] = _upstream.describe()                                                                              # what upstream did with the inputs: items / early_stopped / rolled_out
    unreached_blocks(out)                                                                                               # a package lever whose seam no input reached (zero items / all early-stopped): named, ok
    out["levers"] = lever_states(rep, out)                                                                              # per lever: strategy, state, evidence
    return out


ROLLOUT_SEAM_LEVERS = ("xtr", "dtk", "mkdit")              # the package levers whose verdict fails on `no … call reached … (installed but never ran)`: their seam is the
                                                           # trunk (xtr) / the diffusion roll-out (dtk, mkdit) — unreached when upstream predicted nothing or early-stopped every input


def unreached_blocks(out: dict) -> dict:
    """For each ROLLOUT_SEAM_LEVERS block in the tally that is on, NOT ok, and whose census counts zero calls, in a process where upstream
    rolled out nothing (upstream.no_rollout_reason: ``zero_items`` | ``early_stop``): the block becomes ok with ``reason`` / ``unreached`` =
    ``no_call_reached:<why>`` — the lever engaged and its seam was never reached, by upstream's own decision. Anything else is untouched."""
    from . import upstream as _upstream
    why = _upstream.no_rollout_reason(out.get("upstream"))
    if not why:
        return out
    for name in ROLLOUT_SEAM_LEVERS:
        b = out.get(name)
        if not isinstance(b, dict) or b.get("ok") or not b.get("on", True):
            continue
        calls = (b.get("census") or {}).get("calls")
        if calls is not None and int(calls) == 0:
            b.update(ok=True, reason=f"no_call_reached:{why}", unreached=why)
    return out


def _probe_count(lv, t: dict) -> int:
    """One registry probe against this process's records in the tally ``t`` (registry.py docstring): the count (or 0/1) it names."""
    kind = lv.probe[0]
    if kind == "describe":
        return int((t.get("describe") or {}).get(lv.probe[1]) == lv.probe[2])
    if kind == "stats":
        return int((t.get("HOIST_STATS") or {}).get(lv.probe[1]) or 0)
    if kind == "capture":
        cap = t.get("LAST_CAPTURE") or {}
        return int(bool(cap.get("capture_ms") or cap.get("n_replay")))
    if kind == "mem":
        return int((t.get("mem") or {}).get(lv.probe[1]) or 0)
    if kind == "dtk":
        return int((((t.get("dtk") or {}).get("census") or {}).get(lv.probe[1])) or 0)
    if kind == "mkdit":
        return int((((t.get("mkdit") or {}).get("census") or {}).get(lv.probe[1])) or 0)
    if kind == "xtr":
        return int((((t.get("xtr") or {}).get("census") or {}).get(lv.probe[1])) or 0)
    if kind == "confhoist":
        return int((((t.get("confhoist") or {}).get("census") or {}).get(lv.probe[1])) or 0)
    if kind == "hostlean":
        return int((((t.get("hostlean") or {}).get("census") or {}).get(lv.probe[1])) or 0)
    if kind in ("prefetch", "awrite"):
        return int((((t.get(kind) or {}).get("census") or {}).get(lv.probe[1])) or 0)
    if kind == "graph_levers":                                                                    # warm: engaged iff some capture of the process ran fewer than GRAPH_WARMUP (3) eager steps
        gl = t.get("graph_levers") or {}
        caps, steps = int(gl.get("captures") or 0), int(gl.get("warmup_steps") or 0)
        return int(bool(gl.get(lv.probe[1])) and caps > 1 and steps < 3 * caps)
    if kind == "confln":
        return int(((t.get("confln") or {}).get(lv.probe[1])) or 0)
    if kind == "big":
        return int((_big_per_lever(t).get(lv.probe[1]) or {}).get("ran") or 0)
    if kind == "rowpair":
        return int(bool((t.get("rowpair") or {}).get("installed")) and int(t.get("n_gpu") or 1) > 1)
    ft = t.get("fpf") or {}
    return fpf_probe(lv.name, lv.probe, ft.get("describe") or {}, ft.get("describe_v2") or {})


def _big_per_lever(t: dict) -> dict:
    """The big census per memory lever (``big.state()["census"]["per_lever"]``: ran / skipped / fallback / absent / unexpected unit counts)."""
    return (((t.get("big") or {}).get("census") or {}).get("per_lever")) or {}


def _census(d):
    """A counted-by-reason census as LEVER-line evidence: the adapter's shape keys (``c=64``, ``d=64``) without their ``=``, so every
    ``k=v`` token of the line holds exactly one ``=`` (the core renders the dict as ``k:v,k2:v2``)."""
    return {str(k).replace("=", ""): v for k, v in d.items()} if isinstance(d, dict) and d else None


def _evidence(name: str, t: dict) -> dict:
    """The lever's own numbers from the tally, for its LEVER line (small: counts and the implementing module, never tensors)."""
    if name == "rowpair":
        rp = t.get("rowpair") or {}
        stt = rp.get("state") or {}
        ev = {"n_gpu": t.get("n_gpu"), "sharding": t.get("sharding"), "sites": len(rp.get("sites") or []) or None}
        ev.update({k: stt.get(k) for k in ("layout", "rank", "peak_gib", "rows", "trunk_s") if stt.get(k) is not None})
        sched = stt.get("schedule") or {}
        ev.update({k: sched[k] for k in ("templ_form", "templ_items", "templ_real", "templ_fill_max") if sched.get(k) is not None})   # the template feature's census under n_gpu>1, cumulative over the process's items (row_born; items that reached the template embedder; those with a real template r/items; the most filled tokens of one item)
        ev.update({k: sched[k] for k in ("data_form", "feats_ranks_equal", "feats_digest", "feats_items", "feats_digest_excludes") if sched.get(k) is not None})
        ev.update({k: sched[k] for k in ("apb_core", "apb_core_src", "apb_served", "apb_fallback", "apb_fallback_by", "opm_rows", "opm_rows_source", "opm_row_bytes") if sched.get(k) is not None})
        ev.update({k: sched[k] for k in ("diff_cond_rows", "diff_bias_rows", "diff_q_rows", "diff_bias_cache", "diff_rows_source", "dit_cache_policy", "dit_cache_gib", "dit_free_gib", "dit_cache_fit", "rank_threads", "host_slab", "host_slab_gib", "conf_ztrunk_retire", "pwa_s_chunk", "pwa_qblock", "dit_rows", "dit_bias", "dit_rows_core", "diff_attn_core") if sched.get(k) is not None})
        for _k in ("dit_rows_core", "dit_rows", "dit_bias"):                   # the core's describe strings may carry blanks; lever values are single tokens
            if _k in ev and isinstance(ev[_k], str):
                ev[_k] = "_".join(ev[_k].split())   # n_gpu>1 only: the roll-out's row blocks + pair-bias cache decision (core DiffusionSchedule)
        if stt.get("peak_stage") is not None:                                                                           # n_gpu>1 only: the stage in which this rank's running device maximum last rose + the
            ev["peak_stage"] = stt["peak_stage"]                                                                        # (stage:alloc/max GiB) marks of the last item
            ev["mem_trace"] = ";".join(f"{a}:{b}/{c}" for a, b, c in (stt.get("mem_trace") or [])) or None   # n_gpu>1 only (the schedule census is empty at n_gpu=1): the attention-pair-bias core serving local query rows (sdpa|torch, calls served / named fallbacks) and the OPM row block the MSA module's outer-product statement was sized to   # the input features' form under n_gpu>1: rank 0's by broadcast (rowpair featurise), the digest gate's word, the last item's digest / ordinal
        return ev
    if name.startswith("big_"):
        c = _big_per_lever(t).get(name[len("big_"):]) or {}
        return {"units_ran": c.get("ran", 0), "units_fallback": c.get("fallback", 0), "units_skipped": c.get("skipped", 0), "units_absent": c.get("absent", 0)}
    ft = t.get("fpf") or {}
    d2 = ft.get("describe_v2") or {}
    if name == "graph":
        cap = t.get("LAST_CAPTURE") or {}
        return {"capture_ms": cap.get("capture_ms"), "steps": cap.get("T"), "samples": cap.get("D"), "warmup": cap.get("warmup")}
    if name == "hoist":
        hs = t.get("HOIST_STATS") or {}
        return {"rollouts": hs.get("rollouts", 0), "entries_last": hs.get("entries_last", 0)}
    if name == "graph_safe_ops":
        return {"describe": (t.get("describe") or {}).get("RF3_GRAPH_SAFE_OPS")}
    if name == "mem":
        m = t.get("mem") or {}
        return {"policy": m.get("policy"), "releases": m.get("releases", 0)}
    if name == "fpf_trimul":
        return {"executes_from": ((t.get("kernels") or {}).get("routed") or {}).get("fpf_trimul_v4"), "served": ft.get("served", 0), "fallback": _census(ft.get("fallback")),
                "errors": ft.get("errors", 0)}
    if name == "fpf_gflash":
        return {"executes_from": ((t.get("kernels") or {}).get("routed") or {}).get("flash_triattn"), "calls": (d2.get("counts") or {}).get("triattn") or None}
    if name in ("fpf_ttr",):
        return {"calls": (d2.get("counts") or {}).get("transition") or None}
    if name in ("fpf_apb", "fpf_sapb"):
        return {"calls": (d2.get("counts") or {}).get("apb") or None}
    if name == "fpf_tg":
        tg = d2.get("trunk_graph") or {}
        ev = {"replays": tg.get("replays", 0), "n_graphs": tg.get("n_graphs", 0), "fallbacks": _census(tg.get("fallbacks"))}
        b = ft.get("tg_budget") or {}
        if b.get("on"):
            c = b.get("census") or {}
            ev.update({"budget_max_i": b.get("max_i"), "graphed": c.get("graphed", 0), "skipped": c.get("skipped", 0)})
        return ev
    if name == "fpf_res":
        r = d2.get("res") or {}
        return {"fused": r.get("fused", 0), "unfused": r.get("unfused", 0)}
    if name == "fpf_dattn":
        da = d2.get("dattn") or {}
        return {"calls": da.get("calls", 0), "fallback": da.get("fallback", 0)}
    if name == "dtk":
        dk = t.get("dtk") or {}
        c = dk.get("census") or {}
        ev = {"executes_from": dk.get("impl"), "bias": dk.get("bias"), "served": c.get("served", 0), "fallback": c.get("fallback", 0)}
        if c.get("fallback_by"):
            ev["fallback_by"] = _census(dict(sorted(c["fallback_by"].items())))
        ev.update({"min_tokens": dk.get("min_tokens"), "gated": c.get("gated", 0), "calls": c.get("calls", 0)})
        return ev
    if name == "mkdit":
        mk = t.get("mkdit") or {}
        c = mk.get("census") or {}
        files = ",".join(f"{k.rsplit('/', 1)[-1]}:{str(v)[:12]}" for k, v in sorted((mk.get("files_sha256") or {}).items())) or None
        return {"executes_from": mk.get("impl"), "files": files, "served": c.get("served", 0), "samples": c.get("samples", 0), "hoists": c.get("hoists", 0),
                "min_tokens": mk.get("min_tokens"), "gated": c.get("gated", 0), "calls": c.get("calls", 0)}
    if name == "xtr":
        xk = t.get("xtr") or {}
        c = xk.get("census") or {}
        ev = {"executes_from": xk.get("impl"), "construction": xk.get("construction"), "serve": ",".join(xk.get("serve") or []),
              "served": c.get("served", 0), "routed": c.get("routed", 0), "fallback": c.get("fallback", 0), "calls": c.get("calls", 0)}
        if xk.get("floor") is not None:                                                                                 # a device with a row floor: the floor and its routed calls, by name
            ev["floor"] = xk["floor"]; ev["routed_floor"] = c.get("routed_floor", 0)
        ev["by_key"] = _census(dict(sorted((c.get("by_key") or {}).items())))
        return ev
    if name == "confln":
        c = t.get("confln") or {}
        return {"calls": c.get("calls"), "eps": c.get("eps")} if c else {}
    if name == "confhoist":
        from . import confhoist as _confhoist
        return dict(_confhoist.lever_tokens(t.get("confhoist") or None)) if t.get("confhoist") else {}
    if name == "hostlean":
        from . import hostlean as _hostlean
        return dict(_hostlean.lever_tokens(t.get("hostlean") or None)) if t.get("hostlean") else {}
    if name == "prefetch":
        from . import prefetch as _prefetch
        return dict(_prefetch.lever_tokens(t.get("prefetch") or None)) if t.get("prefetch") else {}
    if name == "awrite":
        from . import awrite as _awrite
        return dict(_awrite.lever_tokens(t.get("awrite") or None)) if t.get("awrite") else {}
    if name == "warm":
        gl = t.get("graph_levers") or {}
        return {"captures": gl.get("captures"), "warmup_steps": gl.get("warmup_steps")} if gl else {}
    return {}


def lever_states(rep, t: dict) -> dict:
    """Per registry lever, this process's state — ``on``: in the mode's set (the memory policy: not ``off``) and its probe shows it acted;
    ``skipped`` with a one-token ``reason``: ``no_fold`` (rf3.graph_flags never imported here), ``below_gate`` (dtk / mkdit: every call under
    its size gate; an FPF kernel lever every call of which its own size gate declined: fpf_tally's ``levers_size_gated``), ``below_floor`` (xtr: every call
    of a served width under the device's row floor), ``probe_zero:<kind>`` (selected, its probe is zero); ``off`` with ``not_in_mode:<mode>``
    or ``serve_only`` — plus its
    strategy id, impl / origin (registry) and its evidence."""
    from .registry import IMPL_OF, LEVERS
    selected = set((rep or {}).get("levers") or [])
    mem_policy = (t.get("mem") or {}).get("policy")
    if mem_policy not in (None, "off"):
        selected.add("mem")                                                   # the memory policy rides every kit row (mem.POLICY), not the row's lever list
    owned = ((rep or {}).get("big") or {}).get("site_owned") or ((t.get("big") or {}).get("site_owned") or {})
    for x in ((t.get("big") or {}).get("levers") or []):
        if x not in owned:                                                    # a lever whose site the row-sharding adapter takes (n_gpu>1) is off by property, below
            selected.add("big_" + x)                                        # the big mode's memory levers (big.LEVERS; registry names carry the big_ prefix)
    ran = bool(t.get("graph_flags_imported"))
    withheld = set((rep or {}).get("withheld") or t.get("withheld") or [])
    out = {}
    for name, lv in LEVERS.items():
        impl, origin = IMPL_OF[name]
        st = {"strategy": lv.strategy, "kit": lv.kit, "impl": impl, "origin": origin}
        if name in withheld:                                                  # MODEL_OPT_LEVERS_OFF: removed from the mode for this run, by name
            st.update({"state": "off", "reason": "withheld"})
        elif name not in selected:
            short = name[len("big_"):] if name.startswith("big_") else name
            st.update({"state": "off", "reason": ("serve_only" if not lv.wired else
                                                  str(owned[short]).split(" — ")[0] if short in owned else      # site_owned:rowpair (n_gpu>1) / site_owned:fpf_ttr (big.ARM_SERVES)
                                                  ("n_gpu:1" if name == "rowpair" and t.get("mode") == "big" else f"not_in_mode:{t.get('mode')}"))})
        elif not ran:
            st.update({"state": "skipped", "reason": "no_fold"})
        elif isinstance(t.get(name), dict) and t[name].get("conflict"):          # a package lever declined by name in this process (dtk under n_gpu>1: rowpair.CONFLICTS)
            st.update({"state": "off", "reason": f"conflict:{t[name]['conflict']}"})
        elif _probe_count(lv, t) > 0:
            st.update({"state": "on"}, **_evidence(name, t))
        elif name in ("dtk", "mkdit") and ((t.get(name) or {}).get("census") or {}).get("calls"):
            st.update({"state": "skipped", "reason": "below_gate"}, **_evidence(name, t))
        elif name in ((t.get("fpf") or {}).get("levers_size_gated") or []):              # every call of the kernel declined by its own size gate (fpf_tally size_gated_levers): routed by name
            st.update({"state": "skipped", "reason": "below_gate"}, **_evidence(name, t))
        elif name == "xtr" and (t.get("xtr") or {}).get("below_floor"):                  # every call of a served width lay below the device's row floor (pf.CARD_ROWS): routed by name
            st.update({"state": "skipped", "reason": "below_floor"}, **_evidence(name, t))
        elif name in ((t.get("fpf") or {}).get("levers_unreached") or []) or (isinstance(t.get(name), dict) and t[name].get("unreached")):
            from . import upstream as _upstream                                              # its seam was never reached: upstream predicted nothing / early-stopped every input
            st.update({"state": "skipped", "reason": f"no_call_reached:{(t.get(name) or {}).get('unreached') or _upstream.no_rollout_reason(t.get('upstream'))}"}, **_evidence(name, t))
        else:
            from . import upstream as _upstream
            why = _upstream.no_rollout_reason(t.get("upstream"))
            st.update({"state": "skipped", "reason": (f"no_call_reached:{why}" if why else f"probe_zero:{lv.probe[0]}")}, **_evidence(name, t))
        out[name] = st
    return out


LEVER_META = ("strategy", "kit", "impl", "origin", "state", "reason")


def lever_lines(t: dict) -> list:
    """One ``LEVER`` line per registry lever — the core's pinned grammar (``opt_core.report.lever_line``): ``[rosettafold3-opt] LEVER
    name=<house name> state=<on|off|skipped> [reason=<token>] impl=<file|kernel module> origin=<kit|core> strategy=<F<k>.x|LOCAL.x>
    <evidence k=v …>``."""
    lines = []
    for name, st in (t.get("levers") or {}).items():
        evidence = [(k, v) for k, v in st.items() if k not in LEVER_META]
        lines.append(_core_report.lever_line(TAG, name, st.get("state"), ("strategy", st.get("strategy")), *evidence,
                                            reason=st.get("reason"), impl=st.get("impl"), origin=st.get("origin")))
    return lines


def print_lever_lines(t: dict) -> None:
    for ln in lever_lines(t):
        _core_report.emit(ln)


def fpf_tally_line(ft: dict) -> str:
    return (f" fpf={ft.get('arm')} served={ft.get('served', 0)} fallback={json.dumps(ft.get('fallback') or {}, sort_keys=True)} errors={ft.get('errors', 0)} "
            f"tg_replays={ft.get('tg_replays', 0)} tg_fallbacks={json.dumps(ft.get('tg_fallbacks') or {}, sort_keys=True)} "
            f"kernel_errors={json.dumps(ft.get('kernel_errors') or {}, sort_keys=True)} levers_acted={_join(ft.get('levers_acted'))} "
            f"levers_silent={_join(ft.get('levers_silent'))} ok={ft.get('ok')}"
            + (f" disengaged_by_property={_join(ft.get('disengaged_by_property'))}" if ft.get("disengaged_by_property") else "")
            + (f" levers_budget_skipped={_join(ft.get('levers_budget_skipped'))}(max_i={(ft.get('tg_budget_skipped') or {}).get('max_i')},"
               f"{','.join(f'{k}={v}' for k, v in sorted(((ft.get('tg_budget_skipped') or {}).get('by_key') or {}).items()))})" if ft.get("levers_budget_skipped") else "")
            + (" levers_unreached=" + ";".join(f"{n}({(ft.get('unreached') or {}).get(n)})" for n in ft["levers_unreached"]) if ft.get("levers_unreached") else "")
            + (" levers_size_gated=" + ";".join(f"{n}({','.join(f'{k}={v}' for k, v in sorted(((ft.get('size_gated') or {}).get(n) or {}).items()))})" for n in ft["levers_size_gated"]) if ft.get("levers_size_gated") else "")
            + (f" reason={ft.get('reason')}" if ft.get("reason") else ""))


def tally_line(t: dict) -> str:
    fpf_s = fpf_tally_line(t["fpf"]) if t.get("fpf") else ""
    if t.get("withheld"):                                                                        # MODEL_OPT_LEVERS_OFF (leversoff.py): the EXIT line names what the run withheld
        from . import leversoff as _leversoff
        fpf_s += _leversoff.token(t["withheld"])
    if t.get("absent"):                                                                          # … and the words it accepted as no-ops (compile=none)
        from . import leversoff as _leversoff
        fpf_s += _leversoff.absent_token(t["absent"])
    if not t.get("graph_flags_imported"):
        return f"{PREFIX} EXIT mode={t.get('mode')}: rf3.graph_flags never imported in this process (no fold ran here)" + fpf_s + f" {ngpu_fields(t)}"
    hs = t.get("HOIST_STATS") or {}
    cap = t.get("LAST_CAPTURE") or {}
    d = t.get("describe") or {}
    up = t.get("upstream") or {}
    from . import upstream as _upstream
    why = _upstream.no_rollout_reason(up) if up else None
    up_s = (f" upstream=items:{up.get('items', 0)},early_stopped:{up.get('early_stopped', 0)},rolled_out:{up.get('rolled_out', 0)}"
            + (f",no_rollout:{why}" if why else "")) if why or int(up.get("early_stopped") or 0) else ""   # named only when upstream skipped or early-stopped something (the line is unchanged otherwise)
    return (f"{PREFIX} EXIT mode={t.get('mode')} rollouts={hs.get('rollouts', 0)} entries_last={hs.get('entries_last', 0)} "
            f"capture={json.dumps(cap, sort_keys=True) if cap else 'none'} describe={json.dumps(d, sort_keys=True)}" + fpf_s + up_s
            + f" {ngpu_fields(t)}")                                                            # the resource axis closes the line


def register_exit_tally(rep: dict, tally_file: "str | None" = None) -> bool:
    """Print the tally at interpreter exit, once per process (the core's registry, ``opt_core.report.register_exit_tally``: the line is
    this module's own grammar, ``tally_line``); ``tally_file`` (the caller's ROSETTAFOLD3_OPT_TALLY_FILE) also gets the same tally as JSON
    from this module's own exit hook, registered after the line's so it runs first. True when registered now."""
    cache: dict = {}

    def tally_of() -> dict:
        if "t" not in cache:
            cache["t"] = tally(rep)
        return cache["t"]

    def levers_and_json():
        print_lever_lines(tally_of())                                                # one LEVER line per registry lever, then (the core's hook) EXIT
        if tally_file:
            rank = os.environ.get("ROWPAIR_RANK", "").strip()                       # the family launcher's rank variable (opt_core.mem.rowpair.launch.ENV_RANK):
            path = tally_file if rank in ("", "0") else f"{tally_file}.rank{rank}"  # rank 0's tally is the run's (pred reads it); a rank r > 0 writes beside it, suffixed
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(tally_of(), fh, indent=1, default=str)
            except OSError as e:
                print(f"{PREFIX} EXIT tally file not written: {e}", file=sys.stderr, flush=True)   # named, never silent: pred reads this file for its verdict
    if not _core_report.register_exit_tally(TAG, lambda: tally_line(tally_of())):
        return False
    atexit.register(levers_and_json)
    return True
