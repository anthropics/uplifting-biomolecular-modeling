"""Observability for af2ig_opt: the activation line, the per-lever evidence lines and the exit tally (all on stderr, prefixed ``[af2ig-opt]``).

* activation — ``ACTIVE mode=<m> route=cli line=<the lever flags as run> levers=<the mode's planned ids> precompile=<N> jax=<v>
  gpu=<name> applied=argv`` (or ``STOCK line=... stripped=...`` for the stock arm, ``DRY-RUN ...`` with the gates' verdicts, or
  ``NOT ACTIVE: <reason>``), formatted from the activation report returned by ``stack.activate()``, before the driver runs;
* lever evidence — one ``LEVER name=<kit id> state=<on|off|skipped> [reason=<why>] impl=<what> origin=<kit|core> strategy=<canonical id> flag=<switch> arm=<mode> [evidence=<record>]``
  line per lever of the registry per arm per process (:func:`lever_lines`, each line by ``opt_core.report.lever_line`` — the tree's one per-lever grammar),
  printed after the run from the driver's own records (``stack.applied``): ``on`` = applied and evidenced, ``skipped`` = in the arm's lever set
  without the driver's evidence (the partial set; its reason), ``off`` = not in this arm's lever set (every lever on the stock arm), or in it and
  superseded by another lever of the arm by the mode's definition (``reason=superseded_by:<lever>_…``: big's L11 once ``-trimul_chunk`` engages);
* peak — one ``PEAK item=<tag> inuse_gib=<GiB>`` line per completed design and one ``PEAK-NOTE scope=pass items=<n> inuse_gib=<max> source=peak_bytes_in_use
  reset=none`` line per process, from the ``design`` records' ``peak_bytes_in_use`` (:func:`peak_lines`; the JAX allocator's in-process peak, process-cumulative), both arms;
* exit — what the driver's process did, printed by the command line after the run (cli.py): items completed against the input count
  (the driver's own score file and checkpoint), the driver's rc, ``partial=<ids|none> allow_partial=on|off``. When nothing ran the tally says so
  (never silent). The exit code and the one partial-activation line before it are the shared core's (``opt_core.report.verdict`` /
  ``partial_line``: ``NOT ACTIVE: partial activation — levers=<ids> (<id>: <why>; …); exit 3 (--allow-partial records and proceeds)`` on a run
  that completed with unevidenced levers, ``PARTIAL allowed: …`` under ``--allow-partial``; a run that failed on its own — driver status ≠ 0 or
  outputs short — names those levers on their LEVER lines and the EXIT line and keeps its own code).
"""
from __future__ import annotations

import os
import sys
from typing import Optional, List

from opt_core import report as _core_report

from . import registry

from . import TAG                                                     # "af2ig-opt"
PREFIX = _core_report.prefix(TAG)                                      # "[af2ig-opt]"
LEVER_STATES = ("on", "off", "skipped")                               # the per-lever evidence words (the tree's convention: opt_core.report.lever_line)
IMPL = {registry.L8: ("opt_core.pallas.serve.attention", "core"), registry.L9: ("subbatch_policy", "core"), registry.L18: ("template_subbatch", "kit"), registry.L19: ("triattn_core_bf16", "core"), registry.L10: ("opt_core.pallas.serve.triangle_attention_block", "core"), registry.L11: ("opt_core.pallas.serve.triangle_multiplication", "core"), registry.L12: ("opm_reassoc_jax", "kit"), registry.TRIMUL: ("rowpair_jax.rowchunk", "core"),
        registry.MEMF: ("xla_client_mem_fraction", "kit"), registry.CCACHE: ("capture.xla_cache", "core"), "L13": ("serialize_executable", "kit"), registry.L15: ("thread_lookahead", "kit"), registry.L16: ("thread_writer", "kit")}            # every other lever: the driver's own code ("predict_pdb", "kit")


def _stack(rep: dict) -> str:
    up = rep.get("upstream") or {}
    gpu = rep.get("gpu") or {}
    return f"jax={up.get('jax') or 'absent'} jaxlib={up.get('jaxlib') or 'absent'} dm-haiku={up.get('dm-haiku') or 'absent'} gpu={gpu.get('name') or 'none'}"


def _gates(rep: dict) -> str:
    """``<gate>=ok|drift|forced|unmet`` per gate (``drift``: the pins gate met by distributions installed at other versions than the pin — named,
    the run proceeds; the activation line's ``note='pins drift …'`` lists them); a ``quiet_ok`` gate (tiles) is named only when unmet, so a
    card the core has rows for prints the same words as a card check without it."""
    g = rep.get("gates") or {}
    def word(v):
        if v.get("ok"):
            return "drift" if v.get("drift") else "ok"
        return "forced" if v.get("forced") else ("aside" if v.get("aside") else "unmet")
    return " ".join(f"{k}={word(v)}" for k, v in g.items() if not (v.get("quiet_ok") and v.get("ok")))


def _ablation(rep: dict) -> str:
    """`` levers_off=<ids>`` right after ``levers=`` when MODEL_OPT_LEVERS_OFF dropped levers of the mode (nothing otherwise: the line is unchanged without the switch)."""
    off = rep.get("levers_off") or []
    return f" levers_off={_core_report.join(off)}" if off else ""


def _compile(rep: dict) -> str:
    """`` compile=stock_jit`` on every kit-mode line (QoL): JAX's jit per distinct length is STOCK behaviour the kit inherits — there is no kit compile lever (L7's AOT
    pass, ccache and the L13 store only shorten that wall). ``--no-compile`` / ``MODEL_OPT_LEVERS_OFF=compile`` asked to drop one: the word says there is none to drop
    (``compile=stock_jit(no-compile:inherent;no_kit_compile_lever)``) — never a refusal, nothing else changes."""
    if rep.get("mode") == "off" or not rep.get("compile"):
        return ""
    return f" compile={rep['compile']}" + ("(no-compile:inherent;no_kit_compile_lever)" if rep.get("no_compile") else "")


def _aside(rep: dict) -> str:
    """`` skipped=<ids>`` after ``levers=`` (and ``levers_off=``) when levers of the mode stepped aside by name in this process (a cache root that cannot be written …)."""
    sk = rep.get("skipped") or {}
    return f" skipped={_core_report.join(list(sk))}" if sk else ""


def _deployment(rep: dict) -> str:
    """`` jit=<on|kept>:<dir>`` / `` jit=skipped:<reason>`` / `` jit=off:levers_off`` right after ``precompile=`` on a kit mode (the ccache lever's placement); nothing on the stock arm."""
    cc = rep.get("ccache")
    if not cc:
        return ""
    from . import ccache as _ccache
    return f" jit={_ccache.word(cc)}"


def _mode_word(rep: dict) -> str:
    """`<mode>` or `<word>→<line>` when the word resolved to another mode's line (``modes.FOLDED``)."""
    f = rep.get("folded_into")
    return f"{rep.get('mode')}→{f}" if f else f"{rep.get('mode')}"


def activation_line(rep: dict) -> str:
    if rep.get("dry_run"):
        if rep.get("gates") is None:                          # nothing resolved: the reason is the whole line
            return f"{PREFIX} DRY-RUN mode={rep.get('mode')} would_refuse={rep.get('would_refuse')!r}"
        s = (f"{PREFIX} DRY-RUN mode={_mode_word(rep)} line={rep.get('line_run') or 'none'} levers={_core_report.join(rep.get('levers_planned'))}{_ablation(rep)}{_aside(rep)} "
             f"precompile={rep.get('precompile')}{_deployment(rep)} {_gates(rep)} {_stack(rep)}{_compile(rep)}")
        if rep.get("would_refuse"):
            s += f" would_refuse={rep['would_refuse']!r}"
        for n in rep.get("notes") or []:
            s += f" note={n!r}"
        return s
    if not rep.get("active"):
        return f"{PREFIX} NOT ACTIVE: {rep.get('reason')}"
    s = (f"{PREFIX} ACTIVE mode={_mode_word(rep)} route={rep.get('route')} line={rep.get('line_run') or 'none'} levers={_core_report.join(rep.get('levers_planned'))}{_ablation(rep)}{_aside(rep)} "
         f"precompile={rep.get('precompile')}{_deployment(rep)} {_stack(rep)}{_compile(rep)} applied={rep.get('applied')}")            # levers= is the mode's PLANNED set (the composition as resolved); what the driver evidenced is on the LEVER lines after the run
    for n in rep.get("notes") or []:
        s += f" note={n!r}"
    return s


def weights_line(rep: dict) -> Optional[str]:
    """`[af2ig-opt] weights=<name> sha256=<12> (pinned)` when AF2_PARAMS holds the pinned parameter file, `[af2ig-opt] weights sha256=<12> not pinned (pinned: <12>) —
    proceeding` when it holds another (the run proceeds either way), either followed by `(cached digest <utc>)` when the
    digest was served from the on-disk memo (`digest_memo.word`); None when no file was digested. The words are stock/check_pins.py's (`weights_words`, the one producer)."""
    from . import stack as _stack
    w = (rep.get("gates") or {}).get("weights") or {}
    from . import digest_memo
    words = _stack.pins_module().weights_words({"present": bool(w.get("sha256")), "name": w.get("name"), "sha256": w.get("sha256"), "pinned": w.get("pinned"), "pin": w.get("pin")})
    return f"{PREFIX} {digest_memo.word(words, w.get('cached_utc'))}" if words else None      # `… (cached digest <utc>)` when the digest came from the memo


def log_activation(rep: dict, stream=None) -> str:
    line = activation_line(rep)
    print(line, file=stream or sys.stderr, flush=True)
    wl = weights_line(rep)
    if wl:
        print(wl, file=stream or sys.stderr, flush=True)
    return line


def _partial_reason(rep: dict, lv: str) -> str:
    return str((rep.get("partial_reasons") or {}).get(lv) or "no reason recorded")


def token(text) -> str:
    """A LEVER-line value: one blank-free token (``opt_core.report.lever_line`` splits a line on blanks) — runs of whitespace become ``_``."""
    return "_".join(str(text).split()) or "none"


def lever_lines(rep: dict) -> List[str]:
    """One ``opt_core.report.lever_line`` per lever of the registry (registry.LEVERS, its order) for the arm that ran, from stack.applied's verdicts: the arm's evidenced
    levers ``on`` (``evidence="<the driver record>"``), its partial levers ``skipped`` (``reason="<why>"``), every other lever ``off`` (the stock arm: all
    of them; a lever of the set superseded by another lever of the arm by the mode's definition — stack.applied's ``superseded`` — with that as its reason,
    ``reason=superseded_by:<lever>_…``); ``name=`` is the kit's lever id (registry.LEVERS), ``strategy=`` the tree's canonical strategy id, then what implements the lever and where that implementation
    lives (``impl= origin=``: ``kit`` for the driver's levers, ``core`` for the shared kernel), the switch that engages it and the arm (``flag= arm=``)."""
    mode = rep.get("mode")
    planned = list(rep.get("levers_planned") or [])
    partial = set(rep.get("partial") or [])
    evidence = rep.get("evidence") or {}
    superseded = rep.get("superseded") or {}
    levers_off = set(rep.get("levers_off") or [])
    skipped = rep.get("skipped") or {}
    out = []
    for lv, lever in registry.LEVERS.items():
        impl, origin = IMPL.get(lv, ("predict_pdb", "kit"))                        # impl = what implements the lever, origin = where that implementation lives (the shared kernel / producer in the core; the driver in the kit)
        ids = (("flag", lever.switch), ("arm", mode))                              # after the pinned head (name=<kit id>, state, [reason], impl, origin, strategy=<canonical id>): the switch, the arm, then the evidence
        if mode == "off" or lv not in planned:
            if mode != "off" and lv in levers_off:                                      # dropped by the ablation switch (or, L19, by the caller's own fp32 word): off, with that as its reason
                why = f"word:{registry.L19_ENV}=fp32" if (lv == registry.L19 and rep.get("l19_word_off")) else f"levers_off:{registry.LEVERS_OFF_ENV}"
                out.append(_core_report.lever_line(TAG, lv, "off", *ids, reason=token(why), impl=impl, origin=origin, strategy=lever.strategy))
            elif mode != "off" and lv in skipped:                                         # stepped aside by name in this process (a cache root that cannot be written, …): skipped with its reason, never partial
                out.append(_core_report.lever_line(TAG, lv, "skipped", *ids, reason=token(str(skipped[lv] or "cannot_run")), impl=impl, origin=origin, strategy=lever.strategy))
            else:
                out.append(_core_report.lever_line(TAG, lv, "off", *ids, reason=token("the stock arm applies no lever" if mode == "off" else f"not in the lever set of {mode} as run"), impl=impl, origin=origin, strategy=lever.strategy))
        elif lv in superseded and lv not in partial:                               # in the set, every call taken by another lever of the mode by its definition: off with that reason — never `on` over zero calls
            out.append(_core_report.lever_line(TAG, lv, "off", *ids, reason=token(superseded[lv]), impl=impl, origin=origin, strategy=lever.strategy))
        elif lv in partial or lv not in evidence:
            out.append(_core_report.lever_line(TAG, lv, "skipped", *ids, reason=token(_partial_reason(rep, lv)), impl=impl, origin=origin, strategy=lever.strategy))
        else:
            out.append(_core_report.lever_line(TAG, lv, "on", *ids, impl=impl, origin=origin, strategy=lever.strategy, evidence=token(evidence[lv])))
    return out


def print_lever_lines(rep: dict, stream=None) -> List[str]:
    lines = lever_lines(rep)
    for line in lines:
        print(line, file=stream or sys.stderr, flush=True)
    return lines


PEAK_FIELD = "peak_bytes_in_use"                                        # the `design` record's field: jax.local_devices()[0].memory_stats()["peak_bytes_in_use"], read by the driver after the design's outputs (predict_pdb.py `_peak_bytes_in_use`, patch 11)


def peak_lines(records) -> List[str]:
    """The per-design allocator peak from the driver's timer records (both arms: the stock line writes the same ``design`` records) — one
    ``<PREFIX> PEAK item=<tag> inuse_gib=<peak_bytes_in_use / 2**30, two decimals>`` line per ``design`` record, ``item=`` the record's ``tag``
    verbatim (the id of the relayed record and of ``<tag>_af2pred.pdb``), then ONE pass line ``<PREFIX> PEAK-NOTE scope=pass items=<n>
    inuse_gib=<max over the items> source=peak_bytes_in_use reset=none``. The value is the JAX allocator's in-process peak on the first local device,
    process-cumulative — JAX has no peak reset, so an item never reads lower than an earlier item of its process, and the note says so (``reset=none``);
    ``nvidia-smi`` shows the preallocated pool (``XLA_PYTHON_CLIENT_MEM_FRACTION``), not this. A ``design`` record without the value (a backend that keeps no
    memory statistics, a driver older than patch 11) gets no item line and is counted on the note, ``unread=<k> reason=memory_stats_unavailable``; a run
    with no ``design`` record reads ``items=0 inuse_gib=0.00 ... reason=no_design_record`` — never silent."""
    designs = [r for r in records if isinstance(r, dict) and r.get("kind") == "design"]
    read = [(str(r.get("tag")), int(r[PEAK_FIELD])) for r in designs if isinstance(r.get(PEAK_FIELD), (int, float)) and not isinstance(r.get(PEAK_FIELD), bool)]
    lines = [f"{PREFIX} PEAK item={tag} inuse_gib={v / 2 ** 30:.2f}" for tag, v in read]
    note = f"{PREFIX} PEAK-NOTE scope=pass items={len(read)} inuse_gib={max((v for _, v in read), default=0) / 2 ** 30:.2f} source={PEAK_FIELD} reset=none"
    if not designs:
        note += " reason=no_design_record"
    elif len(read) < len(designs):
        note += f" unread={len(designs) - len(read)} reason=memory_stats_unavailable"
    return lines + [note]


def print_peak_lines(records, stream=None) -> List[str]:
    lines = peak_lines(records)
    for line in lines:
        print(line, file=stream or sys.stderr, flush=True)
    return lines


def exit_line(mode: str, **fields) -> str:
    body = " ".join(f"{k}={v}" for k, v in fields.items())
    return f"{PREFIX} EXIT pid={os.getpid()} mode={mode} {body}".rstrip()


def print_exit(mode: str, stream=None, **fields) -> str:
    line = exit_line(mode, **fields)
    print(line, file=stream or sys.stderr, flush=True)
    return line
