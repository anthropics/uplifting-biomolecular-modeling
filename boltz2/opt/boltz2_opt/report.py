"""The package's lines on stdout and its exit codes — one place.

    [boltz2-opt] ACTIVE mode=exact route=worker gpu=<name> n_gpu=<P> sharding=<rowpair|none> levers=<a,b,...> fallbacks=<...> worker=<file>+bz2dit kernels=<on|off>
                                                                                                                  the mode is on in the worker this process launched (worker=<file> alone: a base as shipped)
    [boltz2-opt] APPLIED levers=<...> graph=<mode> hoist=<level> ... templates=<none|declared:K,live:J,items:n>   the worker's own evidence, read from its log after it exits
    [boltz2-opt] TEMPLATES record=<id> declared=<k> names=<n> slots=<T> live_slots=<j> real=<j>/<T> [form=row_born n_gpu=<P>]        one line per templated input after APPLIED: the worker's census at the model's entry, echoed (templates.item_lines)
    [boltz2-opt] LEVER name=<lever> state=<on|skipped|off> [reason=<why>] impl=<file|module> origin=<kit|core> strategy=<id> <evidence k=v ...>
                                                                                                                  one line per lever of the mode row after APPLIED, in the core's LEVER grammar (opt_core.report.lever_line;
                                                                                                                  registry.LEVERS strategy / origin), rendered from the worker's evidence
    [boltz2-opt] DRY-RUN mode=<m> route=<r> gpu=<name> n_gpu=<P> sharding=<...> levers=<...>                       check: resolved and gated, nothing applied
    [boltz2-opt] NOT ACTIVE: <reason>                                                                             the mode could not be activated (exit 3, nothing ran silently as stock)
    [boltz2-opt] NOT ACTIVE: partial activation — <detail>; exit 3 (--allow-partial records and proceeds)         a lever of the row fell back (partial_line: pred, warm, check)
    [boltz2-opt] PARTIAL allowed: <detail> (--allow-partial, recorded)                                            the same state accepted by --allow-partial: the run's own exit
    [boltz2-opt] GATE <lever>: <rule>                                                                              a documented gate of the kit, recorded (the exit is the run's own)
    [boltz2-opt] EXIT mode=<m> route=<r> n_gpu=<P> sharding=<...> predictions=<n> ok=<k> failed=<f> rc=<rc> levers_headroom_gated=<graph_sampler,dit_hoist:n|->   the exit tally (atexit), once per process

The `n_gpu=<P> sharding=<rowpair|none>` pair is the core's text (tp.fields: opt_core.mem.ngpu.fields), never spelled in this file; an
`--n_gpu` the kit refuses (P > 1 outside `--mode big`, fewer than P visible GPUs, a P outside tp.SUPPORTED_P) takes the NOT ACTIVE line
with the core's words (tp.check).

Exit codes: 0 ok · 1 failed (a prediction failed, or outputs short: incomplete — an input the stock parser skipped included, named on its
SKIPPED line, and a batch boltz's own predict_step skipped on CUDA out of memory, named on the unit's FAILED line) · 2 usage · 3 not active
(refused, pins not met, kit files missing) or partial activation without --allow-partial.
"""
from __future__ import annotations

import os

import atexit
import sys
from typing import List, Optional

PREFIX = "[boltz2-opt]"
TAG = "boltz2-opt"                                   # opt_core.report.prefix(TAG) == PREFIX
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3
from . import affinity_leg  # noqa: E402  — the affinity leg's ×P word on the LEVER line (standard library at import)
from .kernels import EXIT_KERNELS  # noqa: E402  — 5: the KERNELS REQUIRE guard refused (kernels.py: an accelerator the route expects engaged is absent / fell back)

# The family's partial-activation lines (one formatter, every entry point): the fixed parts are literal, <detail> is this package's own —
# the fallbacks as stack.partial_activation names them (the lever first, then the kit's reason), '; '-joined.
PARTIAL_REFUSED_FMT = "{prefix} NOT ACTIVE: partial activation — {detail}; exit {code} (--allow-partial records and proceeds)"
PARTIAL_ALLOWED_FMT = "{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)"

_TALLY = {"armed": False, "mode": None, "route": None, "n_gpu": 1, "predictions": 0, "ok": 0, "failed": 0, "rc": None, "printed": False, "headroom_gated": 0}
HEADROOM_LEVERS = ("rollout", "dit_hoist")                 # the sampler levers whose memory-headroom gate can send a prediction to the stock sampler (boltz_dit_hoist.headroom_gate)


def _fmt(items) -> str:
    return ",".join(str(x) for x in items) if items else "-"


def ngpu_tokens(n_gpu=1) -> str:
    """``n_gpu=<P> sharding=<rowpair|none>``: the core's positional evidence pair (tp.fields), joined for the ACTIVE / EXIT / DRY-RUN lines.
    Against a core without the n_gpu words the pair reads ``n_gpu=<core_missing:opt_core.mem.ngpu>`` (the run has already been refused by
    name in the gate; the EXIT tally still prints)."""
    from . import tp
    try:
        return " ".join(f"{k}={v}" for k, v in tp.fields(int(n_gpu or 1)))
    except tp.Refused:
        return f"n_gpu=<core_missing:{tp.WORDS_MODULE}>"


def active_line(report: dict) -> str:
    from . import settings                                           # noqa: PLC0415
    return (f"{PREFIX} ACTIVE mode={report.get('mode')} route={report.get('route')} gpu={report.get('gpu') or '-'} {ngpu_tokens(report.get('n_gpu', 1))} "
            f"levers={_fmt(report.get('levers_applied'))} fallbacks={_fmt(report.get('levers_fallback'))} worker={report.get('worker') or '-'} "
            f"{version_tokens()} kernels={report.get('kernels') or '-'} compile={compile_token(report)}"
            + (f" {settings.settings_tokens(report['settings'])}" if report.get("settings") and settings.settings_tokens(report["settings"]) else "")
            + card_off_token(report))


def compile_token(report: dict) -> str:
    """`compile=<word>` of the ACTIVE / DRY-RUN lines (modes.compile_word: off:none_in_kit | off:user); a report may carry its own `compile` value."""
    from .modes import compile_word                                  # noqa: PLC0415
    return report.get("compile") or compile_word()


def version_tokens() -> str:
    """`kit=<boltz2_opt version> opt_core=<shared core version>` of the ACTIVE line (every mode and the xP line print through active_line): which
    tree printed the line, for readers that keep transcripts across versions. `opt_core=absent` when the core does not import."""
    from . import __version__ as kit                                 # noqa: PLC0415
    try:
        from opt_core import __version__ as core                     # noqa: PLC0415
    except Exception:                                                # noqa: BLE001
        core = "absent"
    return f"kit={kit} opt_core={core}"


def card_off_token(report: dict) -> str:
    """`` card_off=<lever>:<reason>[,…]``: levers of the row this card does not run, by name (modes.CARD_DROPS); `` off=<lever>:<reason>[,…]``: the
    row's own ablation entries (modes.MODES[m]["off"]); `` run_off=<lever>:<option>[,…]``: levers this run took off for a boltz predict
    option (modes.RUN_DROPS); `` tp_off=<lever>:<reason>[,…]``: levers the xP line took off at n_gpu > 1 (modes.TP_DROPS). Each token appears only
    when its map is non-empty: a row that runs as composed prints none."""
    from .modes import OFF_KEYS                                      # noqa: PLC0415  (report is imported by modes' importers: call-time)
    out = ""
    for key in OFF_KEYS:
        off = report.get(key) or {}
        if off:
            out += f" {key}=" + ",".join(f"{k}:{v}" for k, v in off.items())
    return out


off_token = card_off_token                                      # the three by-name maps as ACTIVE / DRY-RUN tokens


def _f2(evidence: dict) -> str:
    f2 = evidence.get("f2")
    if f2 is None:
        return "-"
    r = evidence.get("f2_report")
    if not r:
        return str(f2)
    return f"{f2}(flash_calls={r.get('flash_calls', 0)},stock_calls={r.get('stock_calls', 0)},fallbacks={sum((r.get('exceptions') or {}).values())})"


def applied_line(evidence: dict) -> str:
    return (f"{PREFIX} APPLIED levers={_fmt(evidence.get('levers'))} graph={evidence.get('graph_mode') or '-'} hoist={evidence.get('hoist_level') or '-'} "
            f"f2={_f2(evidence)} items={evidence.get('n_items', 0)} n_replay={evidence.get('n_replay') or '-'} "
            f"captured_with_cache={evidence.get('captured_with_cache') if evidence.get('captured_with_cache') is not None else '-'}" + _xl(evidence)
            + f" templates={_templates(evidence)}")


def template_lines(evidence: dict) -> List[str]:
    """The worker's per-input template census (templ_report.per_item, boltz2_opt.templates.item_lines) as this process's own lines: one
    ``[boltz2-opt] TEMPLATES record=<id> … real=<j>/<T>[ form=row_born n_gpu=<P>]`` per input that declared templates (+ the ALL DUMMY
    sentence when none is live) — the words the worker printed at the model's entry, in its transcript, repeated where the caller reads."""
    tr = evidence.get("templ_report") or {}
    items = tr.get("per_item") or []
    if not items:
        return []
    from . import templates                                          # noqa: PLC0415 — words only; nothing is installed by the import
    return [line for r in items for line in templates.item_lines(r)]


def _templates(evidence: dict) -> str:
    """The template guard's census (boltz2_opt.templates.report() as stack.evidence reads it): ``none`` (no input declared a template),
    else ``declared:K,live:J,items:n`` — K declared template entries over the run's items, J live (non-dummy) template slots seen in the
    model's features, n items that declared templates; per-template DROPPED events are the worker log's own lines (evidence.template_dropped)."""
    tr = evidence.get("templ_report") or {}
    if not tr or not tr.get("declared"):
        return "none"
    return f"declared:{tr.get('declared', 0)},live:{tr.get('live', 0)},items:{tr.get('items_checked', 0)}"


def _xl(evidence: dict) -> str:
    """The memory adapter's accounting for the memory rows (stack.evidence: xl_report): `xl=<applied levers> xl_calls=<lever>:<n>,… xl_units=<n> xl_exit=ok|partial|allowed`."""
    xr = evidence.get("xl_report")
    if not xr:
        return ""
    from .stack import XL_ACTIVITY
    st = xr.get("stats") or {}
    applied = list(xr.get("applied") or [])
    calls = ",".join(f"{n}:{st.get(XL_ACTIVITY[n], 0)}" for n in applied if n in XL_ACTIVITY)
    ex = xr.get("exit") or {}
    gate = "-" if not ex else ("allowed" if ex.get("partial") and ex.get("allow_partial") else ("partial" if ex.get("partial") else "ok"))
    return f" xl={_fmt(applied)} xl_calls={calls or '-'} xl_units={xr.get('n_units', '-')} xl_exit={gate}"


def partial_detail(fallbacks: List[str]) -> str:
    return "; ".join(fallbacks)


def partial_reason(fallbacks: List[str]) -> str:
    """The manifest's report.reason for a refused partial activation (the line's <detail> under the same words)."""
    return f"partial activation — {partial_detail(fallbacks)}"


def partial_line(fallbacks: List[str], allow_partial: bool) -> str:
    """The partial-activation line for `fallbacks` (non-empty): refused — ``NOT ACTIVE: partial activation — <detail>; exit 3
    (--allow-partial records and proceeds)`` — or allowed — ``PARTIAL allowed: <detail> (--allow-partial, recorded)``."""
    fmt = PARTIAL_ALLOWED_FMT if allow_partial else PARTIAL_REFUSED_FMT
    return fmt.format(prefix=PREFIX, detail=partial_detail(fallbacks), code=EXIT_NOT_ACTIVE)


def run_exit(report: dict, ev: dict, problems: List[str], fallbacks: List[str], rc: int, n_fail: int, allow_partial: bool) -> int:
    """The exit rule after a run (pred / warm: worker.py) — one place. `problems` (the evidence's, plus the process's exit
    code) or a refused partial activation (a row lever fell back, no --allow-partial) make the report inactive with a reason and exit 3
    (1 when the process itself failed); otherwise the APPLIED line, the allowed-partial / GATE lines, exit 0 (1 when an output is short).
    Records the exit code for the tally and returns it."""
    partial_refused = bool(fallbacks) and not allow_partial
    if problems or partial_refused:
        report["active"] = False
        report["reason"] = "; ".join(problems + ([partial_reason(fallbacks)] if partial_refused else []))
        exit_code = EXIT_NOT_ACTIVE if rc == 0 else EXIT_FAILED
        if problems:
            say(not_active_line(report["reason"]))
        if partial_refused and exit_code == EXIT_NOT_ACTIVE:
            say(partial_line(fallbacks, False))          # the family's line: NOT ACTIVE: partial activation — <detail>; exit 3 (…)
    else:
        say(applied_line(ev)); exit_code = EXIT_OK if n_fail == 0 else EXIT_FAILED
        for line in partial_lines(report):
            say(line)
    for line in template_lines(ev):                                  # the worker's per-input template census, echoed for the caller (both exits)
        say(line)
    if ev.get("has_worker_log"):
        for line in lever_lines(report.get("mode"), ev, report):   # one LEVER line per lever of the row, from the worker's evidence (both exits)
            say(line)
        for line in xfer_lines(ev):                                  # the persistent featurizer's hand-over census, its own line
            say(line)
    set_rc(exit_code)
    return exit_code


def kernels_exit(mode: str, rc: int, records: List[dict], where: str, all_skipped: bool = False) -> int:
    """The KERNELS census's exit rules after a pass — one place, both routes (cli.cmd_pred_off: the stock pass's census record; worker.run:
    the worker passes' records or parsed lines, kernels.parse_line). The child refused in-process (rc 5) -> 5 (its KERNELS-REFUSED
    line names accelerator and route); a pass that ran (rc 0) with no record -> the reader did not stand in the model process: NOT ACTIVE, 3
    (evidence missing); any record whose verdict is not PASS -> the refusal line here and 5 (a contradiction visible only in the totals: an
    off-by-route accelerator that was called, no predict_step) — except the one NO-STEP case that is not about accelerators: ``all_skipped``
    (every requested input was SKIPPED by the stock parser, skipped.py) and every record says NO-STEP: nothing reached predict_step, the
    SKIPPED lines and the outputs-short exit speak, rc unchanged. Otherwise rc unchanged."""
    if rc == EXIT_KERNELS:
        say(f"{PREFIX} KERNELS REQUIRE refused in the {'stock' if mode == 'off' else 'worker'} process (exit {EXIT_KERNELS}: its KERNELS-REFUSED line names accelerator and route)")
        return EXIT_KERNELS
    if rc != 0:
        return rc
    if not records:
        say(not_active_line(f"no KERNELS census ({where}): the kernels reader did not stand in the model process — nothing about the accelerators is proven"))
        return EXIT_NOT_ACTIVE
    bad = [r for r in records if (r or {}).get("verdict") != "PASS"]
    if bad and all_skipped and all((r or {}).get("verdict") == "NO-STEP" for r in records):
        say(f"{PREFIX} KERNELS census verdict=NO-STEP ({where}): no input reached predict_step — every requested input was SKIPPED by the stock parser (named above); nothing about the accelerators to judge")
        return rc
    if bad:
        r = bad[0]
        detail = ", ".join(f"{x.get('accelerator')}={x.get('word')} (expected {x.get('expected')})" for x in (r.get("refused") or [])) or \
                 " ".join(f"{a}={w}" for a, w in (r.get("words") or {}).items())
        say(f"{PREFIX} KERNELS REQUIRE refused: route={r.get('route')} verdict={r.get('verdict')} {detail}; exit {EXIT_KERNELS}")
        return EXIT_KERNELS
    return rc


# ---------------------------------------------------------------- the per-lever census (the core's LEVER grammar) ----------------------------------------------------------------
def _tok(v) -> str:
    """A LEVER value may not contain a blank (a line is split on blanks)."""
    return str(v).replace(" ", "_")


MSA2_BITCMP_KEY = "selftest"                                    # boltz2_opt.msa2's trans2_census key of the run-time bit comparison per shape class {class: {state, compared_calls, compared_elems, …}}


def _kv(d) -> str:
    return ",".join(f"{k}>={v}" for k, v in d.items()) if isinstance(d, dict) else (str(d) if d is not None else "-")


def _r5a2_pairs(classes: dict) -> list:
    """Per class: the compare state, the candidates tested, the undetermined selecting compares, the compared calls and output
    elements — `classes=<class>:<state>:cand<k>/und<u>/calls<c>/elems<e>[:sel=<candidate>];…` — with the per-state class counts before it."""
    cl = {k: v for k, v in (classes or {}).items() if isinstance(v, dict) and "state" in v}
    counts = [(f"classes_{st}", sum(1 for v in cl.values() if v.get("state") == st)) for st in ("proven", "comparing", "floor", "refused")]
    words = ";".join(f"{k}:{v.get('state')}:cand{v.get('candidates_tested', 0)}/und{v.get('undetermined', 0)}/calls{v.get('compared_calls', 0)}/elems{v.get('compared_elems', 0)}" + (f":sel={str(v.get('sel')).replace(' ', '')}" if v.get("sel") is not None else "")
                     for k, v in sorted(cl.items(), key=lambda kv: str(kv[0])))
    return counts + [("classes", words or "-")]


def _superseded(name: str, ev: dict) -> list:
    """The class-level block adapters (trimul / transition / pairblock) print ``superseded_by=pairfuse@c128`` while the PAIRFUSE layer
    driver serves the C=128 stacks of the row (they serve what it hands back: templates C=64, kernels off, small inputs) — the accounting of an
    idle or thinned census, by name."""
    out = []
    pr = ev.get("pairfuse_report") or {}
    if name in ("pairblock", "fused_transition", "fpf_trimul", "fpf_trimul_exact", "xl_trans") and "pairfuse" in (pr.get("applied") or []):
        out.append(("superseded_by", ((ev.get("xl_trans_idle_by") if name == "xl_trans" else None) or "pairfuse@c128")))   # (+templ_skip: the template stack's transitions elided on every pass — stack._templ_all_elided) # xl_trans: the driver's fused transition has no [rows, N, 4C] intermediate to chunk; the lever serves the stacks handed back
    return out


def lever_state(name: str, ev: dict, report: dict) -> tuple:
    """``(state, reason, evidence pairs)`` of one lever of the row from the worker's evidence (stack.evidence) and the run report's partial
    census (levers_fallback / gates): ``on`` = the worker's own record says the lever acted or stands installed behind its declared gate;
    ``skipped`` = selected by the row and not applied (the reason names the record that says so). Nothing here decides the exit: the same
    facts are the problems / fallbacks of stack.evidence and stack.partial_activation."""
    fb = [f for f in (report.get("levers_fallback") or []) if f.split(":")[0].strip() == name]
    if fb:
        return "skipped", _tok(fb[0].split(":", 1)[1].strip() if ":" in fb[0] else fb[0]), []
    if name in ("mask", "resid", "cast", "tln", "mask2", "qkvg"):
        return ("on", None, []) if name in (ev.get("levers") or []) else ("skipped", "not_in_worker_env.boltz_levers", [])
    if name == "rowpair_tp":                                   # the memory mode's n_gpu axis: installed at n_gpu > 1 only (tp_report from every rank's worker)
        tr = ev.get("tp_report") or {}
        if int(report.get("n_gpu") or 1) <= 1:
            return "skipped", "n_gpu=1", []
        if not tr.get("installed"):
            return "skipped", "tp_report.installed_false", []
        calls = tr.get("calls") or {}
        return "on", None, [("n_gpu", tr.get("n_gpu")), ("sharded", ",".join(tr.get("sharded") or []) or "-"), ("replicated", ",".join(tr.get("replicated") or []) or "-"),
                            ("trunk_rows", int(calls.get("trunk_rows", 0))), ("zcond_rows", int(calls.get("zcond_rows", 0))), ("dit_blocks_sharded", int(calls.get("dit_blocks_sharded", 0))), ("conf_rows", int(calls.get("conf_rows", 0))), ("gathers_named", int(calls.get("gathers_named", 0))), ("noise_sync", tr.get("sync_policy")), ("diff_noise", tr.get("diff_noise")), ("rng_bcast_corrected", int(calls.get("rng_bcast_corrected", 0))), ("layers", calls.get("layers", 0)),
                            ("rows_tiled", (tr.get("rows_census") or {}).get("rows_tiled")), ("peak_alloc_gib_ranks", ev.get("tp_peaks") or "-"),
                            ("data_form", ((tr.get("msa_host") or {}).get("source") or {}).get("form") or "-"),       # how the ranks come by their model-input features (the core's DATA_FORMS word; rowpair_msa: rank0_bcast = rank 0 featurized, the other ranks received)
                            ("feats_bcast_gib", ((tr.get("msa_host") or {}).get("source") or {}).get("gib", "-")),     # GiB of feature tensors rank 0 sent per batch (the last batch's)
                            ("entry_inputs_equal", f"{(tr.get('guards') or {}).get('inputs_equal', 0)}/{(tr.get('guards') or {}).get('inputs_checked', 0)}"),   # batches whose feature digest agreed on every rank / batches digested (a disagreement is refused: the run fails by name)
                            ("affinity_leg", affinity_leg.TP_FORM),                                        # upstream's affinity leg on this line: rank 0 runs it, the other ranks receive its exit code and result files
                            ("rng_state_checked", (tr.get("guards") or {}).get("rng_checked", 0)),
                            ("park_z_init", (tr.get("schedule") or {}).get("park_z_init", "-")),           # the ×P placement words as they ACTED (the core's schedule census; modes.TP_EXPORTS sets them)
                            ("msa_host", (tr.get("schedule") or {}).get("msa_host", "-")), ("msa_rows", (tr.get("schedule") or {}).get("msa_rows", "-")),
                            ("msa_m", (tr.get("schedule") or {}).get("msa_m", "-")), ("msa_cols", (tr.get("schedule") or {}).get("msa_cols", "-")), ("msa_m_gib", (tr.get("schedule") or {}).get("msa_m_gib", "-")),
                            ("pwa_s_chunk", (tr.get("schedule") or {}).get("pwa_s_chunk", "-")), ("opm_b_gather_gib", (tr.get("schedule") or {}).get("opm_b_gather_gib", "-")), ("msa_host_cycle_gib", (tr.get("schedule") or {}).get("msa_host_cycle_gib", "-")),
                            ("conf_ztrunk_entry", (tr.get("schedule") or {}).get("conf_ztrunk_entry", "-")), ("conf_ztrunk", (tr.get("schedule") or {}).get("conf_ztrunk", "-")),
                            ("conf_ztrunk_host_gib", (tr.get("schedule") or {}).get("conf_ztrunk_host_gib", "-")), ("conf_pairstack", (tr.get("schedule") or {}).get("conf_pairstack", "-")),
                            ("pairstack_transpose", (tr.get("schedule") or {}).get("pairstack_transpose", "-")),   # ROWPAIR_TRANSPOSE_INPLACE: the pair block's ending-node transpose form as it ran (inplace | p2p)
                            ("msa_z_entry", (tr.get("schedule") or {}).get("msa_z_entry", "-")),                   # ROWPAIR_MSA_PARK_Z: the trunk's z during the MSA module (parked:<where> | device_clone[:why])
                            ("templ_rows", (tr.get("schedule") or {}).get("templ_rows", "ran")),                   # ROWPAIR_TEMPL_SKIP: elided:all_dummy when the pass's template mask was all zero, else the row statement ran
                            ("feats_diet", (tr.get("schedule") or {}).get("feats_diet", "-")), ("feats_diet_saved_gib", (tr.get("schedule") or {}).get("feats_diet_saved_gib", "-")),   # ROWPAIR_FEATS_DIET at the rank-0 source
                            ("dit_rows_core", (tr.get("schedule") or {}).get("dit_rows_core", "-")), ("dit_rows_core_served", (tr.get("schedule") or {}).get("dit_rows_core_served", "-")),
                            ("dit_rows_core_stock", (tr.get("schedule") or {}).get("dit_rows_core_stock", "-")), ("dit_rows_core_maxdiff", (tr.get("schedule") or {}).get("dit_rows_core_maxdiff", "-")),   # ROWPAIR_DIT_CORE: the sharded DiT rows' attention core as it served (apb_attn | torch | unavailable:<why>), calls served / on the statement, first-call max |Δ| vs the fp32 statement
                            ("disto_zT", (tr.get("schedule") or {}).get("disto_zT", "whole")), ("pde_zT", (tr.get("schedule") or {}).get("pde_zT", "whole")),   # ROWPAIR_SYM_ZT: how the (z + z^T) readers came by the transposed rows (blocks | whole)
                            ("zcond_released_gib", (tr.get("schedule") or {}).get("zcond_released_gib", "-")),   # ROWPAIR_ZCOND_RELEASE: the conditioned pair rows freed at the sampler's exit (GiB | kept)
                            ("triatt_stage", (tr.get("schedule") or {}).get("triatt_stage", "-")), ("triatt_stage_planes", (tr.get("schedule") or {}).get("triatt_stage_planes", "-")),
                            ("triatt_stage_aside", (tr.get("schedule") or {}).get("triatt_stage_aside", "-")),   # ROWPAIR_TRIATT_STAGE as it ENGAGED (once:<kernel> planes>0; aside = stepped aside by name)
                            ("errors", len(tr.get("errors") or []))]
    if name == "graph_sampler":
        ok = ev.get("graph_mode") == "graph"
        pairs = [("n_replay", ev.get("n_replay")), ("items", ev.get("n_items", 0)), ("release_min_tokens", ev.get("release_min_tokens")), ("released", ev.get("released")),   # items whose graph + hoist cache were released at sample() return (inputs >= release_min_tokens)
                 ("headroom_gated", ev.get("headroom_gated")), ("projected_gib", (ev.get("headroom_last") or {}).get("projected_gib")), ("free_gib", (ev.get("headroom_last") or {}).get("free_gib"))]   # items the levers' memory-headroom gate sampled on the stock path (the last such item's figures, else the last item's)
        if ok and ev.get("n_items") and ev.get("headroom_gated") == ev.get("n_items"):
            return "skipped", "memory_headroom", pairs                                       # every prediction of this process sampled on the stock path by the gate: engaged, never applied
        return ("on" if ok else "skipped"), (None if ok else _tok(f"worker_env.graph_diffusion={ev.get('graph_mode')}")), pairs
    if name == "dit_hoist":
        ok = str(ev.get("hoist_level") or "") not in ("", "0", "None")
        pairs = [("level", ev.get("hoist_level")), ("captured_with_cache", ev.get("captured_with_cache")), ("cache_released", ev.get("cache_released")), ("headroom_gated", ev.get("headroom_gated")),
                 ("max_tokens", ev.get("sampler_max_tokens") or "-"), ("token_gated", ev.get("token_gated", 0))]
        if ok and ev.get("n_items") and ev.get("headroom_gated") == ev.get("n_items"):
            return "skipped", "memory_headroom", pairs
        if ok and ev.get("n_items") and ev.get("token_gated") == ev.get("n_items"):                 # every prediction of this process above the memory row's token ceiling: engaged, stepped aside by name
            return "skipped", f"above_big_tokens:{ev.get('sampler_max_tokens')}", pairs
        return ("on" if ok else "skipped"), (None if ok else "worker_env.dit_hoist_unset"), pairs
    if name == "flash_triattn":
        pb = ev.get("pairblock_report")
        if pb is not None:                                      # hosted by the core block (fast): the block adapter's report names the flash-core calls it served
            served = (pb.get("census") or {}).get("served") or {}
            ok = "flash_triattn" in (pb.get("applied") or [])
            return ("on" if ok else "skipped"), (None if ok else "not_in_pairblock_report.applied"), [("core", pb.get("variant")), ("flash_calls", sum(n for k, n in served.items() if k != "cueq_below_gate" and k != "cueq")), ("cueq_below_gate", served.get("cueq_below_gate", 0)),
                                                                                                 ("min_tokens", pb.get("min_tokens"))]
        r = ev.get("f2_report") or {}                           # the flash tri-attention patch (big's `flash` base): its own exit counters
        ok = ev.get("f2") is True
        tpx = ((ev.get("tp_report") or {}).get("tpx") or {}).get("triatt") or {}
        tword = str(tpx.get("word") or "")
        if not ok and int(report.get("n_gpu") or 1) > 1 and (tword == "flash_triattn" or tword.startswith("tier:")):   # n_gpu > 1 off the flash patch: the row-sharded trunk's row-block attention core IS the core's fused kernel — the flash kernel by name, or the core's tier door (`tier:<word>`: kernels.triattn's row per row window, stepping aside by name to the flash path) — rowpair TPX word; its census is the core's own LEVER line in the rank transcripts
            return ("on" if tpx.get("state") == "on" else "skipped"), (None if tpx.get("state") == "on" else _tok(tpx.get("reason") or f"rowpair_triatt_{tpx.get('state')}")), [("host", "rowpair_tp"), ("core", tword), ("env", tpx.get("env"))]
        return ("on" if ok else "skipped"), (None if ok else "patch_not_active"), [("flash_calls", r.get("flash_calls", 0)), ("stock_calls", r.get("stock_calls", 0)),
                                                                                  ("fallback", sum((r.get("exceptions") or {}).values())), ("min_tokens", r.get("min_tokens"))]
    if name == "templ_skip":                                        # the dummy-template elision: its own report (census skipped / live / capturing / no_mask / errors)
        tr = ev.get("templskip_report") or {}
        ok = "templ_skip" in (tr.get("applied") or [])
        c = tr.get("census") or {}
        return ("on" if ok else "skipped"), (None if ok else "not_in_templskip_report.applied"), [("served", c.get("skipped")), ("templated_passes", c.get("live")), ("skipped", c.get("skipped")), ("capturing", c.get("capturing")), ("no_mask", c.get("no_mask")), ("errors", c.get("errors"))]
    if name == "pairblock_c64":                                     # the template C=64 stack service of the block's Tier-2 variants (pairblock.KEY_C64 under BOLTZ_PAIRBLOCK_C64=1): the block's report names it in applied; its calls are the x64 shapes
        pb = ev.get("pairblock_report") or {}
        ok = "pairblock_c64" in (pb.get("applied") or [])
        shapes = ((pb.get("census") or {}).get("shapes") or {})
        c64_served = sum(n for k, n in shapes.items() if str(k).split("/")[0].endswith("x64")) if isinstance(shapes, dict) else None
        c64_kept = ((pb.get("census") or {}).get("fallback") or {}).get("kept_out:64x4x32", 0)
        return ("on" if ok else "skipped"), (None if ok else "not_in_pairblock_report.applied"), [("core", pb.get("variant")), ("c64_served", c64_served), ("c64_kept_out", c64_kept), ("min_tokens", pb.get("c64_core_min_tokens") if pb.get("variant") != "cueq" else "-")]
    if name in ("rollout", "align_jacobi64", "align_aligncap", "dit_fused"):        # the sampler adapter (boltz2_opt.sampler.report(): applied / levers / line / gate / module.stats), written by the attach hook [SAMPLER]
        sr = ev.get("sampler_report") or {}; lv = (sr.get("levers") or {}).get(name) or {}; st = ((sr.get("module") or {}).get("stats") or {})
        if name not in (sr.get("applied") or []):
            return "skipped", _tok(lv.get("reason") or "not_in_sampler_report.applied"), []
        pairs = [("word", lv.get("word"))]
        rl = (sr.get("levers") or {}).get("rollout") or {}
        if name == "align_jacobi64" and rl.get("max_tokens") and st.get("calls") and st.get("token_gated") == st.get("calls") and not st.get("samples"):
            return "skipped", f"above_big_tokens:{rl.get('max_tokens')}", pairs + [("rides", "rollout")]
        if name == "rollout":
            cs = st.get("capture_s") or []
            pairs += [("calls", st.get("calls", 0)), ("samples", st.get("samples", 0)), ("replays", st.get("replays", 0)), ("captures", st.get("captures", 0)),
                      ("capture_s_total", round(sum(cs), 3) if isinstance(cs, list) else cs), ("capture_s_max", (max(cs) if cs else 0) if isinstance(cs, list) else "-"), ("capture_failed", st.get("capture_failed", 0)),
                      ("kabsch", st.get("kabsch")), ("predraw", st.get("predraw") or "-"), ("div_recipe", st.get("div_recipe")), ("scope", ",".join(f"{k}:{n}" for k, n in (st.get("scope") or {}).items()) or "-"), ("guard_prints", st.get("guard_prints", 0)),
                      ("max_tokens", lv.get("max_tokens") or "-"), ("token_gated", st.get("token_gated", 0)),      # the memory row's token ceiling (modes.BIG_SAMPLER_CEILINGS) and the calls above it
                      ("gate", "ok" if (sr.get("gate") or {}).get("ok") else _tok((sr.get("gate") or {}).get("reason") or "refused"))]
            if lv.get("max_tokens") and st.get("calls") and st.get("token_gated") == st.get("calls") and not st.get("samples"):   # every call above the ceiling: engaged, stepped aside by name
                return "skipped", f"above_big_tokens:{lv.get('max_tokens')}", pairs
        elif name == "dit_fused":
            ds = (((sr.get("module") or {}).get("dit") or {}).get("stats") or {})
            pairs += [("numerics", "bf16_token_transformer"), ("gemm", ds.get("gemm")), ("attn", ds.get("attn")), ("layers", ds.get("layers")), ("launches_per_layer", ds.get("launches_per_layer")), ("calls", ds.get("calls", 0)),
                      ("served", ds.get("served", 0)), ("scope", ",".join(f"{k}:{n}" for k, n in (ds.get("scope") or {}).items()) or "-"), ("packs", ds.get("packs", 0)),
                      ("pack_gib_last", ds.get("pack_gib")), ("weights_mib", ds.get("weights_mib")), ("weights", ds.get("weights") or "resident"), ("weights_builds", ds.get("weights_builds", "-")),
                      ("host", ("eager" if ((sr.get("levers") or {}).get("rollout") or {}).get("state") != "on" or (st.get("calls") and st.get("token_gated") == st.get("calls"))
                                else "rollout" if not st.get("token_gated") else "rollout+eager")),     # the roll-out's captured step, the stock eager loop (the memory row above its ceiling / without the roll-out), or both in one process
                      ("out_dtype_probe", ds.get("out_dtype_probe")), ("mask_fold", ds.get("mask_fold")), ("gate", "ok" if (sr.get("gate") or {}).get("ok") else _tok((sr.get("gate") or {}).get("reason") or "refused"))]
        return "on", None, pairs
    if name in ("dit_par", "dit_mask", "dit_sba", "dit_smx", "dit_glue"):       # the DITEXACT adapter's report (boltz2_opt.ditexact.report(): applied / refused / words / census / gate) [DITEXACT]
        dr = ev.get("ditexact_report") or {}; c = dr.get("census") or {}; g = dr.get("gate") or {}
        if name in (dr.get("refused") or {}):                     # refused by name at install on this card (unproven_cc:sm_xy): the torch statements serve
            return "off", _tok(str(dr["refused"][name])), [("words", ",".join(dr.get("words") or []) or "-"), ("cc_refused", dict(sorted((c.get("cc_refused") or {}).items())))]
        ok = name in (dr.get("applied") or [])
        pairs = [("words", ",".join(dr.get("words") or []) or "-"), ("installed", c.get("installed", 0)), ("calls", c.get("calls", 0)), ("calls_par", c.get("calls_par", 0)),
                 ("mask_skipped", c.get("mask_skipped", 0)), ("mask_applied", c.get("mask_applied", 0)), ("sba_fused", c.get("sba_fused", 0)), ("sba_torch_by", dict(sorted((c.get("sba_torch_by") or {}).items()))),
                 ("smx_fused", c.get("smx_fused", 0)), ("smx_bitcmp", dict(sorted((c.get("smx_bitcmp") or {}).items()))), ("smx_declined_by", dict(sorted((c.get("smx_declined_by") or {}).items()))),
                 ("glue_fused", c.get("glue_fused", 0)), ("glue_bitcmp", dict(sorted((c.get("glue_bitcmp") or {}).items()))), ("glue_declined_by", dict(sorted((c.get("glue_declined_by") or {}).items()))), ("cc_refused", dict(sorted((c.get("cc_refused") or {}).items()))),
                 ("calls_prev", c.get("calls_prev", 0)), ("class_forward_by", dict(sorted((c.get("prev_by") or {}).items()))), ("capturing_calls", c.get("capturing_calls", 0)),
                 ("n_spath", (dr.get("tune") or {}).get("n_spath")), ("prefetch", (dr.get("tune") or {}).get("prefetch")), ("idle", bool(g.get("idle"))), ("gate", "ok" if g.get("ok") else "refused")]
        return ("on" if ok else "skipped"), (None if ok else "not_in_ditexact_report.applied"), pairs
    if name in ("atom_keys_gather", "atom_glue_hoist", "atom_fused", "atom_gemm"):   # the atom-attention adapter's report (boltz2_opt.atom.report(): applied / units.<unit>.census / gate / gemm) [ATOM]
        ar = ev.get("atom_report") or {}
        ok = name in (ar.get("applied") or [])
        unit = {"atom_keys_gather": "keys", "atom_glue_hoist": "glue", "atom_fused": "fused", "atom_gemm": None}[name]
        if unit is None:
            return ("on" if ok else "skipped"), (None if ok else "not_in_atom_report.applied"), [("word", ar.get("gemm")), ("numerics", f"{ar.get('gemm')}_atom_operands"), ("gate", "ok" if (ar.get("gate") or {}).get("ok") else "refused")]
        c = ((ar.get("units") or {}).get(unit) or {}).get("census") or {}
        mod = ar.get("module") or {}
        return ("on" if ok else "skipped"), (None if ok else "not_in_atom_report.applied"), [("served", c.get("served", 0)), ("fallback", sum((c.get("fallback_by") or {}).values())),
                                                                                          ("fallback_by", dict(sorted((c.get("fallback_by") or {}).items()))), ("errors", sum((c.get("errors") or {}).values())),
                                                                                          ("installed", c.get("installed", 0)), ("refreshes", mod.get("refreshes")),
                                                                                          ("gate", "ok" if (ar.get("gate") or {}).get("ok") else "refused")] + \
               ([("gemm", ar.get("gemm")), ("release", ar.get("release") or "graph"), ("releases", mod.get("releases", 0)), ("resident_mb", (mod.get("last") or {}).get("resident_mb"))] if unit == "fused" else [])
                                                                    # release: the static buffers' lifetime rule (graph: the sampler group's token threshold | sample: every sample() — the memory row); resident_mb after the last sample()
    if name == "graph_trunk":                                       # the trunk CUDA-graph lever: its adapter's report (boltz2_opt.graph.report()); words: captured / replayed / evicted / capture_s [GRAPH]
        ar = ev.get("graph_report") or {}; c = ar.get("census") or {}; g = ar.get("gate") or {}
        ok = "graph_trunk" in (ar.get("applied") or [])
        tot = lambda w: sum(v for u in c.values() for k_, v in u.items() if k_ == w)   # noqa: E731
        eager = {k_: v for u in c.values() for k_, v in u.items() if k_.startswith("eager:")}
        return ("on" if ok else "skipped"), (None if ok else "not_in_graph_report.applied"), [("units", ",".join(ar.get("units") or []) or "-"), ("replayed", tot("replayed")), ("captured", tot("captured")),
                ("eager", sum(eager.values())), ("eager_by", dict(sorted((k_[6:], v) for k_, v in eager.items()))), ("capture_s", ar.get("capture_s_total")), ("pool_gib_max", ar.get("pool_gib_max")),
                ("evicted", ar.get("generations_evicted")), ("min_tokens", ar.get("min_tokens")), ("max_tokens", ar.get("max_tokens")), ("errors", sum((ar.get("errors") or {}).values())),
                ("templ_served", (c.get("templ") or {}).get("served", 0) if "templ" in (ar.get("units") or []) else "-"),   # the template-module unit counts its calls (a hoisted boundary, no graph of its own)
                ("bodies", ",".join(f"{u}:{b}" for u, b in sorted((ar.get("bodies") or {}).items())) or "stock"),          # which body each captured unit bound: stock | pairfuse.pfm_forward / pfnm_forward under BOLTZ_PAIRFUSE (graph BODIES table)
                ("mask_predicate", ar.get("mask_predicate") if any(str(b) != "stock" for b in (ar.get("bodies") or {}).values()) else "-"),
                ("primed_missing", ar.get("primed_missing") if any(str(b) != "stock" for b in (ar.get("bodies") or {}).values()) else "-"),          # a unit whose captured body is another lever's forward (e.g. the PAIRFUSE driver's) names it; stock otherwise
                ("idle", bool(g.get("idle"))), ("gate", "ok" if g.get("ok") else "refused")]
    if name == "pairfuse":                                          # the PAIRFUSE layer driver's own report (boltz2_opt.pairfuse.report(), written by the attach hook); providers= names the cores serving under it [PAIRFUSE]
        pr = ev.get("pairfuse_report") or {}; c = pr.get("census") or {}; g = pr.get("gate") or {}
        ok = "pairfuse" in (pr.get("applied") or [])
        served = c.get("served") or {}
        return ("on" if ok else "skipped"), (None if ok else "not_in_pairfuse_report.applied"), ([("residency", pr.get("residency") or pr.get("variant")), ("numerics", f"{pr.get('residency') or pr.get('variant')}_pair_residency"), ("providers", pr.get("providers")),
                ("served", sum(served.values()) if isinstance(served, dict) else served),
                ("served_by", dict(sorted(served.items())) if isinstance(served, dict) else served), ("layers", c.get("layers")), ("noseq_layers", c.get("noseq_layers")),
                ("fallback", sum((c.get("fallback") or {}).values())), ("fallback_by", dict(sorted((c.get("fallback") or {}).items()))), ("sites", dict(sorted((c.get("sites") or {}).items()))),
                ("mask_real", c.get("mask_real")), ("errors", sum((c.get("errors") or {}).values()))]
                + ([("core_cells", _tok(";".join(f"{k}={v}" for k, v in sorted((c.get("core_cells") or {}).items()))))] if c.get("core_cells") else [])   # a site through the core's provider face: which row / cell served (opt_core.kernels.{triattn,trimul}.describe)
                + ([("handed", _tok(";".join(f"{k}={v}" for k, v in sorted((c.get("handed") or {}).items()))))] if c.get("handed") else [])   # a core pick the provider handed to another row by name at a token count (<site>@<N>:<word>-><row>:<kind>=stacks)
                + ([("core_routes", _tok(";".join(f"{k}={v}" for k, v in sorted((c.get("core_routes") or {}).items()))))] if c.get("core_routes") else [])   # the CUDA rows' served route per (tokens, operand layout)
                + [("idle", bool(g.get("idle"))), ("gate", "ok" if g.get("ok") else "refused")])
    if name in ("waste_chunkcast", "waste_opmmask", "waste_opmdiv", "waste_ctorskip"):   # the WASTE adapter's report (boltz2_opt.waste.report(): applied / module.state / module.stats / gate) [WASTE]
        wr = ev.get("waste_report") or {}; m = wr.get("module") or {}; unit = name[len("waste_"):]
        st_ = (m.get("state") or {}).get(unit) or {}; stats = m.get("stats") or {}; g = wr.get("gate") or {}
        if name not in (wr.get("applied") or []):
            return "skipped", "not_in_waste_report.applied", []
        if st_.get("state") == "skipped":
            return "skipped", _tok(st_.get("reason") or "|".join(st_.get("skipped") or []) or "stepped_aside_by_name"), [("over", dict(m.get("over") or {}))]
        if st_.get("state") != "on":
            return "skipped", _tok(f"waste_state:{st_.get('state')}"), []
        if unit == "chunkcast":
            pairs = [("patched", ",".join(x.split("(")[0] for x in (st_.get("patched") or [])) or "-"), ("skipped", "|".join(st_.get("skipped") or []) or "-"),
                     ("transition_chunked", stats.get("transition_chunked_calls", 0)), ("pwa_chunked", stats.get("pwa_chunked_calls", 0)), ("opm_chunked", stats.get("opm_chunked_calls", 0)),
                     ("delegated", stats.get("transition_delegated_calls", 0) + stats.get("pwa_delegated_calls", 0) + stats.get("opm_delegated_calls", 0)),
                     ("casts_hoisted", stats.get("transition_cast_hoisted", 0) + stats.get("pwa_cast_hoisted", 0) + stats.get("opm_cast_hoisted", 0)), ("over", dict(m.get("over") or {}))]
        elif unit == "opmmask":
            pairs = [("opm_chunked", stats.get("opm_chunked_calls", 0)), ("nummask_computed", stats.get("opm_nummask_computed", 0)), ("nummask_reused", stats.get("opm_nummask_reused", 0)),
                     ("cache_entries", m.get("nummask_entries"))]
        elif unit == "opmdiv":
            pairs = [("opm_chunked", stats.get("opm_chunked_calls", 0)), ("div_out", stats.get("opm_div_out", 0))]
        else:                                                        # ctorskip: first-of-process (model build) — constructor init calls replaced by their RNG draw
            pairs = [("scope", "process_start"), ("calls", stats.get("ctorskip_calls", 0)), ("elements", stats.get("ctorskip_elements", 0))]
        return "on", None, pairs + [("idle", bool(g.get("idle"))), ("gate", "ok" if g.get("ok") else _tok(g.get("reason") or "refused"))]
    if name == "exactln_resid":                                     # the fused residual pass: resid:* served words / LayerNorms the next block took / parked-but-unused, from exactln_report (an available word in no row) [EXACTLN]
        ar = ev.get("exactln_report") or {}; c = ar.get("census") or {}; g = ar.get("gate") or {}; served = c.get("served") or {}
        ok = "exactln_resid" in (ar.get("applied") or [])
        fused = sum(n for k_, n in served.items() if str(k_).startswith("resid:")) if isinstance(served, dict) else 0
        return ("on" if ok else "skipped"), (None if ok else "not_in_exactln_report.applied"), [("fused", fused), ("ln_taken", (served or {}).get("resid_ln_taken", 0) if isinstance(served, dict) else 0),
                                                                                            ("ln_unused", (c.get("fallback") or {}).get("resid_ln_unused", 0)), ("below_min_numel", (c.get("fallback") or {}).get("resid_below_min_numel", 0)), ("gate", "ok" if g.get("ok") else "refused")]
    if name == "msa_pwa_exact":                                     # the exact fused pair-weighted averaging (an available word in no row): its unit state and LEVER line from msa2_report [MSA]
        mr2 = ev.get("msa2_report") or {}; st_ = (mr2.get("units") or {}).get("pwa2x") or {}
        if "pwa2x" not in (mr2.get("applied") or []) or st_.get("state", "on") != "on":
            return "skipped", _tok(st_.get("reason") or "not_in_msa2_report.applied"), []
        c = mr2.get("pwa2_census") or {}; cl = c.get("classes") or {}
        return "on", None, [("mode", "exact"), ("served", c.get("served", 0)), ("compared", c.get("compared", 0)), ("undetermined", c.get("undetermined", 0)), ("calls", c.get("calls", 0)), ("fallback", c.get("fallback", 0)),
                            ("fallback_by", dict(sorted((c.get("fallback_by") or {}).items()))), ("floor", _kv(c.get("floor"))), ("lock", _kv(c.get("lock")))] + _r5a2_pairs(cl)
    if name in ("msa_trans2", "msa_trans2_exact"):                 # the MSA-module levers' adapter report (boltz2_opt.msa2.report(): applied units / units.<unit> state / trans2_census incl. the exact unit's runtime self-check) [MSA]
        mr2 = ev.get("msa2_report") or {}; unit = "trans2" if name == "msa_trans2" else "trans2x"
        st_ = (mr2.get("units") or {}).get(unit) or {}; c = mr2.get("trans2_census") or {}; sc = c.get(MSA2_BITCMP_KEY) or {}
        if unit not in (mr2.get("applied") or []):
            return "skipped", _tok(st_.get("reason") or "not_in_msa2_report.applied"), []
        if st_.get("state", "on") != "on":
            return "skipped", _tok(st_.get("reason") or f"msa2_state:{st_.get('state')}"), []
        pairs = [("mode", st_.get("mode")), ("served", c.get("served", 0)), ("calls", c.get("calls", 0)), ("fallback", c.get("fallback", 0)), ("fallback_by", dict(sorted((c.get("fallback_by") or {}).items())))]
        if unit == "trans2x":
            pairs += [("compared", c.get("compared", 0)), ("floor", _kv(c.get("floor"))), ("lock", _kv(c.get("lock")))] + _r5a2_pairs(sc)   # per (rows, dtype, chunking) class the compare state, calls and output elements compared
        else:
            pairs.append(("numerics", "fused_rounding"))
        return "on", None, pairs
    if name in ("dit_tf32", "dit_bf16", "dit_attn_bf16", "seq_bf16"):   # the PRECISION adapter's report (boltz2_opt.precision.report(): applied / disabled (the `off:` ablation entries) / units.<unit>.{state, census} / gate) [PRECISION]
        pr_ = ev.get("precision_report") or {}; u = (pr_.get("units") or {}).get(name) or {}; c = u.get("census") or {}; g = pr_.get("gate") or {}
        if u.get("state") == "off":                              # the row's ablation entry (BOLTZ_PRECISION=off:<unit>): not installed, named, the stock statement runs
            return "off", "ablation", [("switch", f"BOLTZ_PRECISION=off:{name}")]
        ok = name in (pr_.get("applied") or [])
        served_key = "attn_calls" if name == "seq_bf16" else "calls"
        pairs = [("served", c.get(served_key, 0))] + [(k_, c[k_]) for k_ in sorted(c) if k_ not in (served_key, "facts")] + \
                [(k_, _tok(v_)) for k_, v_ in sorted((c.get("facts") or {}).items())] + [("idle", name in (g.get("idle_units") or [])), ("gate", "ok" if g.get("ok") else _tok(g.get("reason") or "refused"))]
        return ("on" if ok else "skipped"), (None if ok else "not_in_precision_report.applied"), pairs
    if name in ("condproj", "tfeat", "tdummy"):                      # the CONF adapter's report (boltz2_opt.conf.report(): applied / words / census / gate) [CONF]
        cr_ = ev.get("conf_report") or {}; c = (cr_.get("census") or {}).get(name) or cr_.get("census") or {}; g = cr_.get("gate") or {}
        ok = name in (cr_.get("applied") or [])
        pairs = [(k_, c.get(k_)) for k_ in ("calls", "token_lists_fused", "atom_lists_fused", "layers_folded", "featurize_computed", "featurize_reused", "forwards", "elided", "live", "rng_draws_replayed") if isinstance(c, dict) and k_ in c]
        return ("on" if ok else "skipped"), (None if ok else "not_in_conf_report.applied"), pairs + [("gate", "ok" if g.get("ok") else _tok(g.get("reason") or "refused"))]
    if name == "triattn_exact":                                     # the exact word binding of the block's library core (boltz2_opt.triattn_exact.report(): applied / variant / census / gate + facts / evidence): the provider's row for this card and stack, the calls it served, the kernel row's own tally
        xr = ev.get("triattn_exact_report") or {}; c = xr.get("census") or {}; g = xr.get("gate") or {}; f_ = xr.get("facts") or {}; kc = c.get("kernel") or {}
        ok = name in (xr.get("applied") or [])
        by = lambda d: {_tok(k): v for k, v in sorted((d or {}).items())} if d else "-"   # noqa: E731
        pairs = [("word", f_.get("word") or "exact"), ("row", f_.get("row")), ("cc", f_.get("cc")), ("stack", f_.get("stack")), ("at", f_.get("at"))]
        if f_.get("refused"):
            pairs += [("select_refused", f_.get("refused")), ("select_fallback", f_.get("fallback"))]
        pairs += [("calls", c.get("calls", 0)), ("served", c.get("served", 0)), ("refused", by(c.get("fallback"))), ("errors", sum((c.get("errors") or {}).values())),
                  ("rows", by(c.get("rows"))), ("kernel", f"{kc.get('served', 0)}/{kc.get('calls', 0)}"), ("kernel_refused", by(kc.get("refused"))),
                  ("idle", bool(g.get("idle"))), ("gate", "ok" if g.get("ok") else "refused")]
        return ("on" if ok else "skipped"), (None if ok else "not_in_triattn_exact_report.applied"), pairs
    if name in ("pairblock", "fused_transition", "exactln"):        # the block adapters' reports and the layer_norm replica's (exactln_report has the same shape: applied / variant / census / gate) [EXACTLN]
        key = {"pairblock": "pairblock_report", "fused_transition": "transition_report", "exactln": "exactln_report"}[name]
        ar = ev.get(key) or {}; c = ar.get("census") or {}; g = ar.get("gate") or {}
        ok = name in (ar.get("applied") or [])
        served = c.get("served") or {}
        if name == "exactln":                                       # the layer_norm replica's card facts and first-call bit comparison per width class (selftest: served / mismatched classes)
            f_ = ar.get("facts") or {}; stt = ar.get("selftest") or c.get("selftest") or {}
            extra_ = [("cc", f_.get("cc")), ("cc_proven", f_.get("cc_proven")), ("bitcmp_classes", len(stt) if isinstance(stt, dict) else stt),
                      ("bitcmp_mismatch", sum(n for w, n in (c.get("fallback") or {}).items() if str(w).startswith("bitcmp_"))), ("provider", f_.get("provider"))]   # provider: kit | core:opt_core-<v> (variant on | core)
        else:
            extra_ = []
        return ("on" if ok else "skipped"), (None if ok else f"not_in_{key}.applied"), [("variant", ar.get("variant")), ("served", sum(served.values()) if isinstance(served, dict) else served),
                                                                                     ("fallback", sum((c.get("fallback") or {}).values())), ("fallback_by", dict(sorted((c.get("fallback") or {}).items())))] + \
                                                                                    [(k_, ar[k_]) for k_ in ("min_rows", "max_rows", "piece_rows", "pieces", "trailing") if ar.get(k_) is not None] + \
                                                                                    [("errors", sum((c.get("errors") or {}).values())), ("idle", bool(g.get("idle"))), ("gate", "ok" if g.get("ok") else "refused")] + \
                                                                                    _superseded(name, ev) + extra_ + \
                                                                                    ([("core_cells", _tok(";".join(f"{k}={v}" for k, v in sorted((ar.get("core_cells") or {}).items()))))] if ar.get("core_cells") else [])   # min_rows / max_rows / piece_rows: the card's served row rule (the adapter's CARD_ROWS, boltz2_opt.rowfloor), on a card that has one, + pieces / trailing of the last call the card pieced; superseded_by: the layer driver serving the stack instead; core_cells: a `core.<word>` variant's provider row / cell (opt_core.kernels.triattn.describe)
    # (exactln adds its card facts and run-time bit-comparison summary below)
    if name in ("fpf_trimul", "fpf_trimul_exact"):
        tr = ev.get("trimul_report") or {}; c = tr.get("census") or {}; t_ = tr.get("tier") or {}   # tier: the provider tier word's routes (word / row / cell / routes)
        ok = name in (tr.get("applied") or [])
        if not ok and (tr.get("disabled") or {}).get(name):            # the row's provider word named a row this card / stack cannot serve: refused at install, by name
            return "skipped", _tok(tr["disabled"][name]), [("provider_word", tr.get("provider_word"))]
        return ("on" if ok else "skipped"), (None if ok else "not_in_trimul_report.applied"), [("served", c.get("served", 0)), ("fallback", sum((c.get("fallback") or {}).values())),
                                                                                             ("fallback_by", dict(sorted((c.get("fallback") or {}).items()))), ("min_tokens", tr.get("min_tokens"))] + \
                                                                                            [(k_, tr[k_]) for k_ in ("max_tokens", "provider_word") if tr.get(k_) is not None] + \
                                                                                            ([("provider", _tok(c.get("provider"))), ("kernel", _tok(c.get("kernel")))] if tr.get("provider_word") else []) + \
                                                                                            ([("word", t_["word"]), ("row", _tok(t_.get("row") or "-")), ("cell", _tok(t_.get("cell") or "-")),
                                                                                              ("routes", _tok("face:" + (",".join(f"{k}={n}" for k, n in sorted(t_["face"].items())) or "0")
                                                                                                              + ";aside:" + (",".join(f"{k}={n}" for k, n in sorted((t_.get("aside") or {}).items())) or "0")))] if t_ else []) + \
                                                                                            [("idle", bool((tr.get("gate") or {}).get("idle"))), ("gate", "ok" if (tr.get("gate") or {}).get("ok") else "refused")] + _superseded(name, ev)   # max_tokens: the row's ceiling (above it the engine's TriMul by name); provider_word / provider / kernel: the core provider's row + cell serving by the row's word
    if name in ("fpf_opm", "fpf_pwa"):
        mr_ = ev.get("msa_report") or {}; unit = "opm" if name == "fpf_opm" else "pwa"
        u = (mr_.get("units") or {}).get(unit) or {}; c = u.get("census") or {}
        ex_ = (mr_.get("execution") or {}).get(unit)
        if ex_ == "replaced_by_rowpair":                              # n_gpu > 1: the row statements own the computation (rowpair.REPLACED_LEVERS); not installed, named
            return "skipped", "replaced_by_rowpair", [("n_gpu", mr_.get("n_gpu")), ("execution", "replaced_by_rowpair"), ("statement", _tok(u.get("statement") or "rowpair.REPLACED_LEVERS"))]
        ok = name in (mr_.get("applied") or [])
        return ("on" if ok else "skipped"), (None if ok else "not_in_msa_report.applied"), [("execution", ex_ or f"executed:{c.get('calls', 0)}"), ("served", c.get("served", 0)), ("fallback", c.get("fallback", 0)),      # census = opt_core.counters.Ledger.fields(): fallback is the count, fallback_by the reasons
                                                                                       ("errors", sum((c.get("errors") or {}).values()) if isinstance(c.get("errors"), dict) else c.get("errors", 0)), ("cfg", u.get("cfg")), ("tma", u.get("tma"))]
    if name == "writer_overlap":                             # the background writer's own census (boltz2_opt.writer.report(), written by the attach hook)
        w = (ev or {}).get("writer_report") or {}
        if not w:
            return "skipped", "no_writer_report", []
        if not w.get("installed"):
            return "skipped", _tok(str(w.get("disposition") or "not_installed").replace("skipped_by_name:", "")), [("execution", _tok(str(w.get("disposition") or "not_installed")))]
        f = [("backend", w.get("backend")), ("items", w.get("items", 0)), ("served", w.get("served", 0)), ("inline", w.get("inline", 0))]
        if w.get("inline_by"):
            f.append(("inline_by", ",".join(f"{k}:{v}" for k, v in sorted(w["inline_by"].items()))))
        f.append(("fallback", w.get("fallback", 0)))
        if w.get("fallback_by"):
            f.append(("fallback_by", ",".join(f"{_tok(str(k))}:{v}" for k, v in sorted(w["fallback_by"].items()))))
        f += [("failed", w.get("failed", 0)), ("moves", w.get("moves", 0)), ("queued_max", w.get("queued_max", 0)), ("join_s", w.get("join_s")), ("bytes_written", w.get("bytes_written", 0)),
              ("helper_pid", w.get("helper_pid")), ("forked_before_cuda", w.get("forked_before_cuda")), ("gate", "ok" if (w.get("gate") or {}).get("ok") else "refused"),
              ("execution", _tok(f"dead:{w['dead']}") if w.get("dead") else f"executed:{w.get('served', 0)}")]
        return "on", None, f
    if name == "prefetch":                                   # the persistent featurizer's own census (boltz2_opt.prefetch.report(), written by the attach hook)
        pr = (ev or {}).get("prefetch_report") or {}
        if not pr:
            return "skipped", "no_prefetch_report", []
        if not pr.get("installed"):
            return "skipped", _tok(str(pr.get("disposition") or "not_installed").replace("skipped_by_name:", "")), [("execution", _tok(str(pr.get("disposition") or "not_installed")))]
        fields = [("served", pr.get("served", 0)), ("fallback", pr.get("fallback", 0)), ("fallback_by", dict(sorted((pr.get("fallback_by") or {}).items()))),
                  ("forks_avoided", pr.get("forks_avoided", 0)), ("ref_forks", pr.get("ref_forks", 0)), ("helper_lineage", pr.get("helper_lineage", "zygote")),
                  ("forked_before_cuda", pr.get("forked_before_cuda")), ("bit_compare", pr.get("bit_compare")), ("pin", pr.get("pin")),
                  ("featurize_s_total", pr.get("featurize_s_total")), ("gate", "ok" if (pr.get("gate") or {}).get("ok") else "refused")]
        if pr.get("refused"):
            return "refused", _tok(str(pr["refused"])), fields
        return "on", None, fields
    from .registry import MEMORY_LEVERS
    if name in MEMORY_LEVERS:                                    # the memory adapter's report (boltz2_opt.big.report(): applied, stats, settings, env, the record's exit gate)
        from .stack import XL_ACTIVITY
        xr = ev.get("xl_report") or {}; st = xr.get("stats") or {}; xs = (xr.get("settings") or {}).get(name) or {}
        ok = name in (xr.get("applied") or [])
        partial = name in ((xr.get("exit") or {}).get("partial") or [])
        if name in XL_ACTIVITY:
            pairs = [("calls", st.get(XL_ACTIVITY[name], 0))] + [(k, xs[k]) for k in sorted(xs)]
            if name == "xl_free":
                pairs.append(("freed_gib", round((st.get("free_bytes") or 0) / 2**30, 2)))
            if name == "relpos_lazy":
                pairs += [("released_gib", round((st.get("relpos_lazy_bytes") or 0) / 2**30, 2)), ("recomputed", st.get("relpos_lazy_recomputed", 0))]
        else:
            pairs = [("conf", (xr.get("env") or {}).get("PYTORCH_CUDA_ALLOC_CONF")), ("effective", (xr.get("env") or {}).get("alloc_effective"))]
        if partial:
            pairs.append(("census", "partial"))
        pairs += _superseded(name, ev)                              # xl_trans under the PAIRFUSE layer driver: superseded_by=pairfuse@c128 (it serves the stacks the driver hands back)
        return ("on" if ok else "skipped"), (None if ok else "not_in_xl_report.applied"), pairs
    return "skipped", "no_evidence_rule", []


def lever_lines(mode: str, ev: dict, report: dict) -> List[str]:
    """One LEVER line per lever of the mode row (modes.levers), in the core's grammar (opt_core.report.lever_line): ``impl`` / ``origin`` /
    ``strategy`` from registry.LEVERS (the canonical strategy ids of opt_core/STRATEGIES.json or this kit's LOCAL.boltz2.<name>), the state and
    the evidence pairs from lever_state."""
    from opt_core.report import lever_line
    from . import modes, registry
    out = []
    row = modes.resolve(mode)
    names = list(row["levers"]) + (["rowpair_tp"] if int(report.get("n_gpu") or 1) > 1 else [])   # the n_gpu axis's lever rides the row at P > 1 only
    for name in names:
        L = registry.LEVERS[name]
        state, reason, pairs = lever_state(name, ev, report)
        fields = {k: (_tok(v) if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in pairs}
        out.append(lever_line(TAG, name, state, *fields.items(), reason=reason, impl=L["impl"], origin=L["origin"], strategy=L["strategy"]))
    for name, why in modes.off_levers(row).items():                     # every lever that left the row BY NAME — the card's drops, the row's ablation entries, the run's drops (modes.RUN_DROPS): one state=off line each, never silent
        L = registry.LEVERS[name]
        out.append(lever_line(TAG, name, "off", reason=why, impl=L["impl"], origin=L["origin"], strategy=L["strategy"]))
    return out


def xfer_lines(ev: dict) -> List[str]:
    """``[boltz2-opt] XFER lever=prefetch word=… [min_mib=…] feats_gib_max=… files_read=… gib_read=… helper=… helper_timeout_s=… loader_timeout_s=…``
    — the persistent featurizer's hand-over census (boltz2_opt.xfer: a CPU storage of ``min_mib`` MiB or more travelled as a disk
    file, not a /dev/shm segment) on its OWN line after the LEVER lines, whose token sets are unchanged (downstream tools parse
    them). Empty when the worker log carries no transport census (the lever not installed)."""
    pr = (ev or {}).get("prefetch_report") or {}
    xf = pr.get("xfer") if isinstance(pr.get("xfer"), dict) else None
    if not pr.get("installed") or not xf:
        return []
    words = [f"word={_tok(str(xf.get('word') or 'not_installed'))}"] + ([f"min_mib={xf.get('min_mib')}"] if xf.get("word") == "auto" else [])
    words += [f"feats_gib_max={pr.get('feats_gib_max', 0)}", f"files_read={xf.get('files_read', 0)}", f"gib_read={xf.get('gib_read', 0)}", f"read_s={xf.get('read_s', 0)}",
              f"helper={_tok(str(xf.get('helper_word') or 'unknown'))}", f"helper_timeout_s={xf.get('helper_timeout_s')}", f"loader_timeout_s={xf.get('loader_timeout_s')}"]
    if xf.get("errors"):
        words.append(f"errors={len(xf['errors'])}")
    return [f"{PREFIX} XFER lever=prefetch " + " ".join(words)]


def partial_lines(report: dict) -> list:
    """After the APPLIED line: the accepted fallbacks (one PARTIAL allowed line, --allow-partial recorded) and one line per documented
    gate — named, never silent."""
    fb = list(report.get("levers_fallback") or [])
    out = [partial_line(fb, True)] if fb else []
    out += [f"{PREFIX} GATE {k}: {v}" for k, v in (report.get("gates") or {}).items()]
    return out


def dry_run_line(report: dict) -> str:
    return (f"{PREFIX} DRY-RUN mode={report.get('mode')} route={report.get('route')} gpu={report.get('gpu') or '-'} {ngpu_tokens(report.get('n_gpu', 1))} "
            f"levers={_fmt(report.get('levers'))} compile={compile_token(report)} env={' '.join(f'{k}={v}' for k, v in (report.get('env') or {}).items()) or '-'}" + card_off_token(report))


def route_line(name: str, r: dict) -> str:
    """`[boltz2-opt route] <name> served from <core copy> runtime_imports=<...>` — the kernel route as held before the launch.
    Its own prefix, deliberately: not one of the lines a log reader counts or reads."""
    ri = ",".join(f"{k}={'core' if v and v.startswith(os.path.dirname(r['core_copy'])) else (v or 'none')}" for k, v in sorted((r.get("runtime_imports") or {}).items())) or "-"
    return f"{PREFIX.replace(']', ' route]')} {name} served from {r['resolved']} runtime_imports={ri}"


def not_active_line(reason: str) -> str:
    return f"{PREFIX} NOT ACTIVE: {reason}"


def say(line: str, file=None) -> None:
    print(line, file=file or sys.stdout, flush=True)


def arm_tally(mode: Optional[str], route: Optional[str], n_gpu: int = 1) -> None:
    """Print the EXIT tally once when the process ends (idempotent); a later call updates the fields (the accepted n_gpu after the gate)."""
    _TALLY.update(mode=mode, route=route, n_gpu=int(n_gpu or 1))
    if not _TALLY["armed"]:
        _TALLY["armed"] = True
        atexit.register(_print_tally)


def count(ok: int = 0, failed: int = 0) -> None:
    _TALLY["predictions"] += ok + failed; _TALLY["ok"] += ok; _TALLY["failed"] += failed


def set_rc(rc: int) -> None:
    _TALLY["rc"] = rc


def tally_headroom(gated: int) -> None:
    """Predictions the sampler levers' memory-headroom gate sampled on the stock path (stack.evidence ``headroom_gated``), for the EXIT line."""
    _TALLY["headroom_gated"] += int(gated or 0)


def headroom_word() -> str:
    """``levers_headroom_gated=graph_sampler,dit_hoist:<n>`` when the gate acted on n predictions of this process, ``levers_headroom_gated=-`` otherwise."""
    n = int(_TALLY.get("headroom_gated") or 0)
    return f"levers_headroom_gated={','.join(HEADROOM_LEVERS)}:{n}" if n else "levers_headroom_gated=-"


def tally_line() -> str:
    return (f"{PREFIX} EXIT mode={_TALLY['mode']} route={_TALLY['route']} {ngpu_tokens(_TALLY['n_gpu'])} predictions={_TALLY['predictions']} ok={_TALLY['ok']} "
            f"failed={_TALLY['failed']} rc={_TALLY['rc'] if _TALLY['rc'] is not None else '-'} {headroom_word()}")


def _print_tally() -> None:
    if _TALLY["armed"] and not _TALLY["printed"]:
        _TALLY["printed"] = True
        say(tally_line())


def tally_snapshot() -> dict:
    return {k: _TALLY[k] for k in ("mode", "route", "predictions", "ok", "failed", "rc", "headroom_gated")}


def reset_tally() -> None:  # tests only
    _TALLY.update(armed=False, mode=None, route=None, n_gpu=1, predictions=0, ok=0, failed=0, rc=None, printed=False, headroom_gated=0)
