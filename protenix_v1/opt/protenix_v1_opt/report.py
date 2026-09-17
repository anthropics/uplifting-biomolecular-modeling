"""The package's lines on stderr and its exit rule — every format string and the one exit-code table live here (the single source the
tests and any reader of the logs use).

Lines: `[protenix-v1-opt] ARMED ...` once at activation, `[protenix-v1-opt] KERNELS ...` right after it (the routed kernels' resolved paths), `[protenix-v1-opt] ACTIVE ... partial=<levers|->` once the levers are on the
runner, `[protenix-v1-opt] NOT ACTIVE: <reason>` on every refusal (one formatter, `not_active_line`), and on a partial run
`[protenix-v1-opt] NOT ACTIVE: partial activation — <levers>: <the kit's own reason>; exit 3 (--allow-partial records and proceeds)` —
or, with the allowance, `[protenix-v1-opt] PARTIAL allowed: <levers>: <reason> (--allow-partial, recorded)`. In the memory mode,
`[protenix-v1-opt] MEMORY mode=big <setting>=<value>(was <before>) ...` (memory_line) after the ACTIVE line: the upstream's inference
settings the package applied (modes.MEMORY_PRESET) and the values they replaced. After a run whose arm
carries `gflash`, `[protenix-v1-opt] FLOOR gflash min_tokens=<floor> items_atleast=<n> items_below=<n> served=<fused calls> state=<served|floor-off|partial>`
(floor_line): the flash triangle-attention token floor (modes.TRIATTN_FLOOR) against the run's items — `floor-off` = every item below the
floor, the stock attention core by design (not a fallback, not partial), followed by `NOTE gflash inactive by size: …` (floor_note; the ACTIVE
line of a gflash mode ends `smalln=gflash:cueq_core(N<floor)`); `partial` = an item at or above the floor and no fused call. After any run,
`[protenix-v1-opt] DEGRADED <lever> <path>=<state> why=<reason> ...` (degraded_line) once per lever whose kernel served every call but on a
slower path of identical numerics — one such path: `gflash fast_launch=disabled` (opt_core/kernels/flash_triattn.py: the cached
direct launcher switched itself off or failed and the kernel runs through plain Triton JIT launches; `fast_launch_stats()`); a named event
(the DEGRADED line), never silent, not a fallback to stock and not partial. Last, one
`[protenix-v1-opt] LEVER name=<strategy id> lever=<kit word> state=<on|off|skipped> [reason=…] k=v …` line per lever of the kit (lever_lines:
the shared per-lever grammar, opt_core.report prefix + kv; STRATEGY_IDS maps the kit's lever words to the shared F<k>.<name> strategy ids): the
mode's levers with their counters, the levers the mode does not compose as `off`, and the package's `memory` / `det` rows. On a card where
an exact-class pair lever (`gblock`, `xtr`) has a served row range (the shared core's served-row rule, opt_core.attn.pair_fused exact_rows: compute capability 8.0), its LEVER line also
carries `min_rows=<n> max_rows=<n> range=<served|range-off|partial>` (range_state): calls outside the range run the stock forward by name
(`gated_by=stock:below_min_rows:<k>` / `stock:above_max_rows:<k>`), and a run whose every c=128 pair call fell outside idles the lever —
`range-off`, `served=0`, `state=on`, not partial; on a card without a range (9.0) nothing is added to the line. A run whose every item sits at
or below a lever's size gate (the trimul levers at N_token <= 100, the transition levers under 4096 pair rows: report.gate_state) prints that
lever `state=skipped reason=below_gate … gate=<n<=100|rows<4096> n=<largest item>` and exits 0 — the stock statement by design, accounted, not partial.
`log_exit_lines()` prints the exit rule's line, FLOOR, DEGRADED and the LEVER lines in that order — the one call every verb makes after a run.

Exit codes (opt_core.report, one per condition, every verb): EXIT_OK 0; EXIT_FAIL 1 (the stock CLI's own failure, or an item that failed / never
returned inside `predict` — an out-of-memory names itself on the `ITEM event=failed` line; the partial rule is then not judged: `NOTE exit rule not judged …`); EXIT_USAGE 2;
EXIT_NOT_ACTIVE 3 — a gate refused, the kit refused an arm, or the activation was PARTIAL: a lever of the mode with no evidence of application in the kit's own
account (`levers_ptx1.describe()`: `counts`, the sampler's and the hoist's counters) or a fallback the kit took after the
lever was engaged — unless `--allow-partial` records the allowance, in which case the exit is the
run's own. `kit_evidence()` reads the kit's counters lever by lever (the counter names are the kit's, cited per lever); `verdict()` is
the rule; `partial_exit_line()` / `partial_allowed_line()` the two lines.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from opt_core import report as CORE
from opt_core.report import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE   # noqa: F401  the one exit table (opt_core.report)

TAG = "protenix-v1-opt"
ABLATED = "ablated"                                     # the LEVER line's reason for a lever MODEL_OPT_LEVERS_OFF withheld from the mode for this run (ablation.REASON)

# The strategy id of every lever of the kit — canonical ids of the shared vocabulary (opt_core/STRATEGIES.json: F<k>.<name>; LOCAL.* =
# engine-specific), keyed by the kit's own lever word (modes.LEVERS / the trimul word): the `name=` and `strategy=` of the per-lever LEVER line.
STRATEGY_IDS = {
    "exact": "F2.fpf_trimul_exact",         # FPF v3 EXACT triangle multiplication (opt_core/kernels/fpf_trimul)
    "fast": "F2.fpf_trimul_fast",           # fpf_trimul_v4 generic Triton cell (opt_core/kernels/fpf_trimul_v4)
    "gblock": "F1.triatt_lean_exact",       # the core's fused triangle-attention block around the stock cuEquivariance core (opt_core.attn.pair_fused, impl fpf)
    "gflash": "F1.flash_triatt",            # the same block around the core's flash_triattn cell (opt_core.attn.pair_fused + opt_core/kernels/flash_triattn)
    "tricuda": "F1.flash_triatt",           # gflash's core through the provider by the `fast` tier word (opt_core.kernels.triattn)
    "triexact": "F1.flash_triatt",          # gblock's core through the provider by the `exact` tier word (opt_core.kernels.triattn: row triattn_exact where vouched, the library op by name elsewhere)
    "xtr": "LOCAL.fused_transition",        # the core's fused pair transition on the stock LayerNorm output (opt_core.attn.pair_fused, impl fpf)
    "ttr": "LOCAL.fused_transition",        # the core's fused LN+SwiGLU transition (opt_core.attn.pair_fused, impl fpf / lnl)
    "sg": "F3.cuda_graph_sampler",          # graphed diffusion denoiser step
    "hoist": "LOCAL.step_invariant_hoist",  # DiT pair-bias hoist out of the sampler loop
    "summary_hostidx": "F6.host_sync_elimination",  # summary-confidence chain bookkeeping from one host copy of the ids (lib/ptx1_summary_host.py)
    "ditattn": "F5.flash_attn_dense",        # fused flash attention + shared pair bias on the DiT token modules (ptxfpf/apb_ptx1.py over opt_core.kernels.apb row fpf_apb)
    "ditattnfp16": "F4.autocast_policy",     # the precision lever on ditattn (fp16 operands)
    "atomattn": "F5.flash_attn_dense",       # the atom transformer's local-window attention kernel
    "dit_attn_exact": "LOCAL.protenix_v1.dit_attn_exact",  # the DiT pair-bias attention on the shared core's bit-exact sm_90 kernel (opt_core.kernels.apb row dit_exact)
    "template_dedupe": "LOCAL.protenix_v1.template_dedupe",  # template-embedder distinct-template evaluation (ptxfpf/ptx1_templ.py)
    "tmpl_triatt": "LOCAL.protenix_v1.tmpl_triatt",          # the template Pairformer's triangle attention through the fused block (ptxfpf/ptx1_templ.py)
    "tmpl_xtr": "LOCAL.protenix_v1.tmpl_xtr",                # the template Pairformer's pair transition through pair_fused (exact construction) (ptxfpf/ptx1_templ.py)
    "tmpl_pairfused": "LOCAL.protenix_v1.tmpl_pairfused",    # the template tri-attention block's fpf prologue/epilogue with LN fused (ptxfpf/ptx1_templ.py)
    "tmpl_trimul": "LOCAL.protenix_v1.tmpl_trimul",          # the template Pairformer's triangle multiplication through opt_core.trimul.by_word (ptxfpf/ptx1_templ.py)
    "tmpl_trimul_exact": "LOCAL.protenix_v1.tmpl_trimul_exact",   # the same route under the provider's exact tier word (--mode exact)
    "sampler_prep": "LOCAL.protenix_v1.sampler_prep",        # the graphed sampler's host path (infopt_graphs.protenix.sampler_prep)
    "lazy_init": "F6.weights_residency_init",    # start-up: the dead random init skipped at model construction (lib/ptx1_lazy_init.py)
    "keep_pool": "LOCAL.protenix_v1.keep_pool",  # caching-allocator policy: the stock in-forward empty_cache sites skipped (lib/ptx1_keep_pool.py)
    "memory": "F7.chunked_eval",           # the memory mode: the upstream's chunked pair path (modes.MEMORY_PRESET)
    "det": "F4.det_recipe",                 # the deterministic recipe (`--det 1`)
    "pfattn": "LOCAL.protenix_v1.pfattn",     # the trunk / confidence pairformer single attention with pair bias on the fused producer + flash core (ptxfpf/trunk2_ptx1.py over lib/protenix_fpf_apb)
    "opm_fused": "LOCAL.protenix_v1.opm_fused",   # the MSA module's OuterProductMean on the fused MSA kernels (ptxfpf/trunk2_ptx1.py over lib/protenix_fpf_msa)
    "pwa_fused": "LOCAL.protenix_v1.pwa_fused",   # the MSA module's MSAPairWeightedAveraging on the fused MSA kernels (ptxfpf/trunk2_ptx1.py over lib/protenix_fpf_msa)
    "cond_dedupe": "LOCAL.protenix_v1.cond_dedupe",   # DiffusionConditioning once per step on the sample-invariant noise level (ptxfpf/ditfast_ptx1.py over lib/protenix_fpf_ditfast)
    "dit_fused": "LOCAL.protenix_v1.dit_fused",       # the 24-block token DiffusionTransformer as one fused forward (lib/protenix_fpf_ditfast dit_fast)
    "dit_lowp": "F4.autocast_policy",                 # dit_fused's fp16 precision word
    "atom_fused": "LOCAL.protenix_v1.atom_fused",     # both 3-block atom transformers as fused stacks (lib/protenix_fpf_ditfast atom_fast)
    "atom_attn_exact": "LOCAL.protenix_v1.atom_attn_exact",   # the atom local-window attention by the shared apb face's exact tier word (ptxfpf/ditfast_ptx1.py over opt_core.kernels.apb)
}
PREFIX = CORE.prefix(TAG)
ALLOW_PARTIAL_FLAG = "--allow-partial"

# the partial grammar is the core's (opt_core.report.PARTIAL_REFUSED / PARTIAL_ALLOWED: `{prefix} NOT ACTIVE: partial activation — {detail};
# exit 3 (--allow-partial records and proceeds)` and `{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)`); the detail is the kit's
PARTIAL_EXIT_REASON_FMT = CORE.PARTIAL_REFUSED.split(" NOT ACTIVE: ", 1)[1]                              # the reason part: what stack passes to ActivationError
PARTIAL_ALLOWED_FMT = CORE.PARTIAL_ALLOWED
PARTIAL_DETAIL_FMT = "{levers}: {reason}"                                                                 # the lever names first, then the kit's own reason


def gpu_label(gpu) -> str:
    """`gpu='<name>' sm=<cc>` — this package's ACTIVE/ARMED token pair (downstream log parsers read `gpu=.+? sm=<cc>`);
    opt_core.report.gpu_label renders the one-token `name(smNN)` form, a different grammar, so this renderer stays."""
    if not gpu or "name" not in gpu:
        return "gpu=none"
    return f"gpu={gpu['name']!r} sm={gpu['sm']}"


def armed_line(rep: dict) -> str:
    return (f"{PREFIX} ARMED mode={rep.get('mode')} arm={rep.get('arm')} line={rep.get('line')} tier={rep.get('tier')} det={int(bool(rep.get('det')))} "
            f"protenix={rep.get('protenix_version')} {gpu_label(rep.get('gpu'))} package={rep.get('package_version')}{ablated_token(rep)}{capped_token(rep)} "
            f"(the levers apply when the stock runner is built)")


def capped_token(rep: dict) -> str:
    """`` capped=<lever,…>@<cap>`` when the run's largest input put the sampler-graph levers above the graph's token cap (modes.graph_cap:
    ``rep["levers_capped"]``, ``rep["graph_cap"]``); the empty string otherwise."""
    capped = rep.get("levers_capped") or ()
    if not capped:
        return ""
    return f" capped={','.join(capped)}@{(rep.get('graph_cap') or {}).get('cap')}"


def ablated_token(rep: dict) -> str:
    """`` ablated=<lever,…>`` when ``MODEL_OPT_LEVERS_OFF`` withheld levers of the mode from this run (``rep["levers_ablated"]``, the
    ablation module's token, request order); the empty string otherwise, so an un-ablated run's lines read exactly as before."""
    from . import ablation as A
    return A.token(rep.get("levers_ablated") or ())


def kernels_line(rep: dict) -> str:
    """`[protenix-v1-opt] KERNELS <name>=<resolved path> ... (opt_core <version>)`: where each routed kernel resolves (stack.route_kernels)."""
    routed = (rep.get("kernels") or {}).get("routed") or {}
    from opt_core import __version__ as core_version
    return f"{PREFIX} KERNELS " + " ".join(f"{name}={v['resolved']}" for name, v in routed.items()) + f" (opt_core {core_version})"


def activation_line(rep: dict) -> str:
    cfg = rep.get("cfg") or {}
    on = ",".join(k for k, v in cfg.items() if v is True)
    lv = rep.get("levers") or {}
    smp = lv.get("sampler") or {}
    partial = ",".join(rep.get("partial") or []) or "-"
    return (f"{PREFIX} ACTIVE mode={rep.get('mode')} arm={rep.get('arm')} trimul={cfg.get('trimul')} levers={on or '-'} "
            f"sampler_graphs={smp.get('graphs')} hoist={smp.get('hoist_installed')} fastln={(smp.get('fastln') or {}).get('how')} "
            f"det={int(bool(rep.get('det')))} partial={partial} {ngpu_fields(rep)} {gpu_label(rep.get('gpu'))}{smalln_token(rep)}{ablated_token(rep)}{capped_token(rep)}")


def smalln_token(rep: dict) -> str:
    """` smalln=gflash:cueq_core(N<floor)` at the END of the ACTIVE line of a mode that carries `gflash` (the floor the package exported,
    rep['triattn_floor']): below the flash triangle-attention floor the fused block serves every call around the stock cuEquivariance attention
    core — a served route decided by size up front, counted `gflash:cueq<floor`, never a partial activation. Empty for every other mode."""
    floor = (rep.get("triattn_floor") or {}).get("exported")
    if not isinstance(floor, int) or not (rep.get("cfg") or {}).get("gflash"):
        return ""
    return f" smalln=gflash:cueq_core(N<{floor})"


def ngpu_fields(rep_or_p) -> str:
    """`n_gpu=P sharding=<rowpair|none>` — the core's token text (opt_core.mem.ngpu via ngpu.fields), from a report dict or a P."""
    from . import ngpu
    p = rep_or_p.get("n_gpu", 1) if isinstance(rep_or_p, dict) else rep_or_p
    return ngpu.fields(p if p is not None else 1)


def exit_line(mode: str, rc, n_gpu=1) -> str:
    """`[protenix-v1-opt] EXIT mode=<m> rc=<code|stock> n_gpu=P sharding=<rowpair|none>` — the last line of every verb (rc=stock on the
    environment route, where the exit code is the stock CLI's)."""
    return f"{PREFIX} EXIT mode={mode} rc={rc} {ngpu_fields(n_gpu)}"


def log_armed(rep: dict, stream=None) -> str:
    return log(armed_line(rep), stream)


def log_activation(rep: dict, stream=None) -> str:
    return log(activation_line(rep), stream)


def log_kernels(rep: dict, stream=None) -> str:
    return log(kernels_line(rep), stream)


def memory_line(rep: dict) -> str:
    """`[protenix-v1-opt] MEMORY mode=<mode> <key>=<value>(was <before>) ...`: the memory mode's inference settings as applied to the
    stock runner's configs (big.apply, the engine adapter), once per process after the ACTIVE line."""
    m = rep.get("memory") or {}
    parts = " ".join(f"{k}={v!r}(was {m.get('replaced', {}).get(k)!r})" for k, v in (m.get("applied") or {}).items())
    return f"{PREFIX} MEMORY mode={rep.get('mode')} {parts}"


def log_memory(rep: dict, stream=None) -> str:
    return log(memory_line(rep), stream)


def floor_line(evidence: Dict[str, dict]) -> Optional[str]:
    """`[protenix-v1-opt] FLOOR gflash min_tokens=<floor> items_atleast=<n> items_below=<n> served=<n> state=<served|floor-off|partial>` from
    the verdict's evidence; None for an arm without `gflash`, a run with no kit account, or an unknown floor."""
    g = evidence.get("gflash") or {}
    f = g.get("floor")
    if not f:
        return None
    return (f"{PREFIX} FLOOR gflash min_tokens={f['min_tokens']} items_atleast={f['items_atleast']} items_below={f['items_below']}"
            f"{' items_unsized=' + str(f['items_unsized']) if f.get('items_unsized') else ''} served={int(g.get('served') or 0)} state={f['state']}")


def floor_note(evidence: Dict[str, dict]) -> Optional[str]:
    """`[protenix-v1-opt] NOTE gflash inactive by size: …` when the run's floor state is `floor-off` (every item below the floor): the fused
    block served around the stock cuEquivariance attention core, counted, named — not a partial activation; None otherwise."""
    f = (evidence.get("gflash") or {}).get("floor")
    if not f or f.get("state") != "floor-off":
        return None
    served = int((evidence.get("gflash") or {}).get("served") or 0)
    return note_line(f"gflash inactive by size: every item is below the flash triangle-attention floor (N_token < {f['min_tokens']}, items_below={f['items_below']}); "
                     f"the fused block served {served} calls around the stock cuEquivariance attention core (gflash:cueq<floor) — a served route, not a partial activation")


def log_floor(v: dict, stream=None) -> Optional[str]:
    line = floor_line(v.get("evidence") or {})
    if not line:
        return None
    out = log(line, stream)
    note = floor_note(v.get("evidence") or {})
    if note:
        log(note, stream)
    return out


def lever_impl(lever: str) -> Tuple[str, str]:
    """(impl, origin) of a lever: `impl` = the head of its registry file entry (modes.LEVERS[lever]["file"]: the module or package that
    implements it), `origin` = core when that head is under opt_core, else kit; the package's own rows: memory (the upstream's inference
    settings), det (det.py)."""
    own = {"memory": ("infer_setting", "kit"), "det": ("det.py", "kit")}
    if lever in own:
        return own[lever]
    from . import modes as M                      # modes imports kit, not report: no cycle
    entry = (getattr(M, "LEVERS", {}) or {}).get(lever) or {}
    head = re.split(r"[\s,(]", str(entry.get("file") or "unknown").strip(), 1)[0].rstrip("/") or "unknown"
    return head, ("core" if head.startswith("opt_core") else "kit")


def _token(v) -> str:
    """A kv value with no blank in it (the LEVER line is split on blanks by its readers)."""
    return re.sub(r"\s+", "_", str(v))


def lever_line(lever: str, state: str, reason: Optional[str] = None, **facts) -> str:
    """`[protenix-v1-opt] LEVER name=<strategy id> state=<on|off|skipped> [reason=<token>] impl=<module> origin=<core|kit> [k=v ...] lever=<kit word>`
    — the shared per-lever line (opt_core.report.lever_line: the pinned key order and grammar; one line per lever per process). This
    wrapper supplies the kit's data only: the strategy id (STRATEGY_IDS), impl / origin (lever_impl), values with no blank inside
    (`_token`), and the kit's own lever word as the trailing key."""
    impl, origin = lever_impl(lever)
    sid = STRATEGY_IDS.get(lever, lever)                  # canonical per opt_core/STRATEGIES.json (the core refuses an alias or an unknown id)
    pairs = [(k, _token(CORE.kv(("x", v)).split("=", 1)[1])) for k, v in facts.items()] + [("lever", lever)]
    return CORE.lever_line(TAG, sid, state, *pairs, reason=(_token(reason) if reason else None), impl=impl, origin=origin, strategy=sid)


def _counter_facts(e: dict) -> dict:
    """The evidence keys of a mode lever in the grammar's order: served, fallback (+fallback_by), gated (+gated_by), retried (+retried_by),
    then its floor state / row range, then extras (degraded, buf_release)."""
    f = {"served": int(e.get("served") or 0)}
    for k in ("fallback", "gated", "retried"):
        d = e.get(k) or {}
        if d:
            f[k] = sum(int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 1 for v in d.values()); f[k + "_by"] = d
    if e.get("floor"):
        f["min_tokens"] = e["floor"]["min_tokens"]; f["floor"] = e["floor"]["state"]
    if e.get("rows"):                                                            # an exact-class pair lever's served row range on this card (range_state)
        if "min_rows" in e["rows"]:                                                # a served row range (the core's exact_rows rule: gblock / xtr); the template levers' by-design idle state carries `why` instead
            f["min_rows"] = e["rows"]["min_rows"]; f["max_rows"] = e["rows"]["max_rows"]
        f["range"] = e["rows"]["state"]
        if e["rows"].get("why"):
            f["range_why"] = e["rows"]["why"]
    if (e.get("gate") or {}).get("state") == "gate-off":                          # the lever's size gate held for every item (gate_state): its word and the run's largest item
        f["gate"] = e["gate"]["word"]; f["n"] = e["gate"]["n_max"]
    if (e.get("ceiling") or {}).get("state") == "ceiling-off":                    # every item above the tri-attention ceiling (ceiling_state): the ceiling and the run's smallest item
        f["ceiling"] = e["ceiling"]["tokens"]; f["n"] = e["ceiling"]["n_min"]
    for k in ("construct_s", "patched", "recheck", "parts", "poison", "cell", "named"):              # lazy_init's facts (construction seconds, init functions no-op'd, recheck verdict); sampler_prep's (parts word, poison verdict); the apb adapter's (cell = <operands>@<cc of the cell that served>, named = the by-name reason a cell of another card / none serves)
        if e.get(k) is not None:
            f[k] = e[k]
    if (e.get("card") or {}).get("state") == "card-off":                          # no build of the lever for this card / stack: the package's own reason
        f["card"] = e["card"]["why"]
    if e.get("subsumed_by"):                                                       # a sampler attention lever whose site a fused stack serves (kit_evidence): its calls are the stack's
        f["subsumed_by"] = e["subsumed_by"]
    if (e.get("aside") or {}).get("state") == "aside" and not e.get("row") and "aside" not in f:     # a fused-sampler word the package refused by name (ditfast_ptx1): the reason token
        f["aside"] = e["aside"]["word"]
    for k, v in sorted((e.get("wfacts") or {}).items()):                          # the fused-sampler words' own facts (ditfast_ptx1.word_account: blocks, act, lowp, routes, ...)
        f[k] = v
    if e.get("row"):                                                             # a provider word (core_evidence): the row it binds, the lever it rides, its by-design state, its facts, the selection per key
        f["row"] = e["row"]
        if e.get("served_by"):
            f["served_by"] = ",".join(f"{k}:{v}" for k, v in e["served_by"].items())
        if e.get("host"):
            f["host"] = e["host"]
        if e.get("aside"):
            f["aside"] = e["aside"]["word"]
        for fact in e.get("facts") or ():
            k, _, v = str(fact).partition("=")
            if k and v:
                f[k] = v
        if e.get("selections"):
            f["selections"] = ";".join(str(x) for x in e["selections"])
    else:
        for fact in e.get("facts") or ():                                        # a word without a provider row carrying its own `key=value` facts (gflash: cores=; xtr / ttr: word=<tier>)
            k, _, v = str(fact).partition("=")
            if k and v and k not in f:
                f[k] = v
        if e.get("selections") and "selections" not in f:                       # the transition levers' provider Selection lines (levers_ptx1 TRANSITION: one per lever / family / shape)
            f["selections"] = ";".join(str(x) for x in e["selections"])
    if e.get("degraded"):
        f["degraded"] = sorted(e["degraded"])
    br = e.get("buf_release")
    if isinstance(br, dict) and br.get("state"):
        f["buf_release"] = br["state"]; f["buf_released"] = int(br.get("released") or 0)
    return f


def lever_lines(evidence: Dict[str, dict], rep: Optional[dict] = None) -> List[str]:
    """One LEVER line per lever of the kit after a run: the mode's levers from the verdict's evidence (kit_evidence: served / fallback /
    gated / retried counters, the floor / row-range state where the lever has one — `on` when it served or sits in its by-design
    floor-off / range-off state, `skipped` with a one-token reason otherwise), the levers the mode does not compose as `off`, and the package's own
    rows from the report: `memory` (the memory preset), `alloc` (the opt-in allocator policy), `det` (the recipe)."""
    rep = rep or {}
    mode = rep.get("mode")
    ablated = tuple(rep.get("levers_ablated") or ())      # MODEL_OPT_LEVERS_OFF: withheld from the mode for this run, written `off reason=ablated` (ablation.REASON)
    capped = tuple(rep.get("levers_capped") or ())        # the sampler-graph levers above the graph's token cap for this run's largest input (modes.graph_cap): `skipped reason=above_cap`, exit 0
    gc = rep.get("graph_cap") or {}
    out = []
    for lv, e in evidence.items():
        facts = _counter_facts(e)
        by_design = (e.get("floor") or {}).get("state") == "floor-off" or (e.get("rows") or {}).get("state") == "range-off"
        if e.get("fallback"):
            out.append(lever_line(lv, "skipped", reason="fallback", **facts))
        elif not e.get("served") and (e.get("gate") or {}).get("state") == "gate-off":   # every item at or below the lever's size gate: accounted, exit 0 (report.partial_of skips it)
            out.append(lever_line(lv, "skipped", reason="below_gate", **facts))
        elif not e.get("served") and (e.get("ceiling") or {}).get("state") == "ceiling-off":   # every item above the tri-attention ceiling: accounted, exit 0 (report.partial_of skips it)
            out.append(lever_line(lv, "skipped", reason="above_ceiling", **facts))
        elif not e.get("served") and (e.get("card") or {}).get("state") == "card-off":   # no build for this card / stack: the lever stepped aside by name, accounted, exit 0
            out.append(lever_line(lv, "skipped", reason="card_off", **facts))
        elif not e.get("served") and (e.get("knob") or {}).get("state") == "knob-off":   # a stock kernel / dtype knob the user set routed the run off this lever's path: accounted, exit 0
            out.append(lever_line(lv, "skipped", reason="stock_knob:" + e["knob"]["flag"], **facts))
        elif not e.get("served") and (e.get("aside") or {}).get("state") == "aside":     # a provider word that served no call by design (its row refused by name on this card / dtype, the library threshold, host lever absent, no core call): accounted, exit 0; `aside=<word>` names it
            out.append(lever_line(lv, "skipped", reason="aside", **facts))
        elif not e.get("served") and not by_design:
            out.append(lever_line(lv, "skipped", reason="served0", **facts))
        else:
            out.append(lever_line(lv, "on", **facts))
    composed = set(evidence)
    for lv in ("exact", "fast", "gblock", "gflash", "triexact", "tricuda", "xtr", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "dit_attn_exact", "ditattn", "ditattnfp16", "atomattn", "lazy_init", "template_dedupe", "tmpl_triatt", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused", "pfattn", "opm_fused", "pwa_fused", "sampler_prep", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact", "tmpl_trimul_exact"):
        if lv in ablated:
            out.append(lever_line(lv, "off", reason=ABLATED))
        elif lv in capped:                                # stepped aside by name above the cap: the stock sampler loop serves (modes.GRAPH_CAP_LEVERS)
            from . import modes as _M                    # (report is imported by modes' importers; read at call time)
            out.append(lever_line(lv, "skipped", reason=_M.CAP_REASON, served=0, gated_by=f"graph_cap:{gc.get('cap')}", n=gc.get("n_token")))
        elif lv not in composed and not (lv in ("exact", "fast") and composed & {"exact", "fast"}):
            out.append(lever_line(lv, "off", reason=f"not_in_mode:{mode}"))
    from . import big as B                              # the memory mode's lever rows in the core's grammar for them (big.lever_rows: applied / refused / off by flag or ablated, property gate or not adopted); [] outside big; chunk_pair's settings are the `memory` row below
    out += B.lever_rows(mode, ablated=ablated)
    mem = rep.get("memory")
    if "chunk_pair" in ablated:                           # the memory preset is chunk_pair's setting: withheld with it
        out.append(lever_line("memory", "off", reason=ABLATED))
    else:
        out.append(lever_line("memory", "on", **{k.rsplit(".", 1)[-1]: v for k, v in (mem.get("applied") or {}).items()}) if mem else
                   lever_line("memory", "off", reason=f"not_in_mode:{mode}"))
    dr = rep.get("det_report")
    out.append(lever_line("det", "on", det=1, det_scatter=int(bool(dr.get("det_scatter_active"))), warn_only=int(bool(dr.get("warn_only")))) if dr else
               lever_line("det", "off", reason="det0"))
    return out


def log_levers(v: dict, rep: Optional[dict] = None, stream=None) -> List[str]:
    return [log(line, stream) for line in lever_lines(v.get("evidence") or {}, rep)]


def degraded_lines(evidence: Dict[str, dict]) -> List[str]:
    """`[protenix-v1-opt] DEGRADED <lever> <path>=<state> why=<reason> [<counter>=<n> ...]` — one line per lever of the mode whose evidence
    carries `degraded` (kit_evidence): the lever served, on a slower path of identical numerics, and the record names it."""
    out = []
    for lv, e in evidence.items():
        for path, d in (e.get("degraded") or {}).items():
            extra = " ".join(f"{k}={v}" for k, v in d.items() if k not in ("state", "why"))
            out.append(f"{PREFIX} DEGRADED {lv} {path}={d.get('state')} why={d.get('why')!r}" + (f" {extra}" if extra else ""))
    return out


def log_degraded(v: dict, stream=None) -> List[str]:
    return [log(line, stream) for line in degraded_lines(v.get("evidence") or {})]


def log_exit_lines(v: dict, rep: Optional[dict] = None, stream=None) -> None:
    """Every line the package prints after a run, in order: the exit rule's (log_verdict), FLOOR (log_floor),
    DEGRADED (log_degraded), one LEVER line per lever (log_levers; `rep` = the activation report: mode, memory, alloc, det), the memory
    mode's census line (log_big_exit), then the row-sharded pair stack's LEVER line (log_rowpair), the TEMPLATE exit line (log_templates),
    then the ITEMS census (log_items)."""
    log_verdict(v, stream); log_floor(v, stream); log_degraded(v, stream); log_levers(v, rep, stream); log_big_exit(stream); log_rowpair(stream)
    log_templates(stream); log_items((rep or {}).get("items_census"), stream)


def log_big_exit(stream=None) -> List[str]:
    """The memory mode's census line after the LEVER lines (big.exit_lines: opt_core.mem.record.exit_line on the census gate — one line
    naming every partial unit / refused lever, nothing on a complete run); [] outside the mode."""
    from . import big as B
    return [log(line, stream) for line in B.exit_lines()]


def log_rowpair(stream=None) -> str:
    """The row-sharded pair stack's LEVER line (rowpair.lever_line: the family's producer; on with the layout facts at n_gpu>1, off
    reason=n_gpu_1 at n_gpu=1)."""
    from . import rowpair
    line = log(rowpair.lever_line(), stream)
    rowpair.emit_kernel_lines()                               # the core prints its row-kernel LEVER lines here (none when the line ran its torch statements)
    return line


def item_failed_line(rec: dict) -> str:
    """`[protenix-v1-opt] ITEM event=failed item=<name> N_token=<n> error=<ExceptionType:_text>` — printed when a `predict` item raises (the stock's
    handler then logs the traceback and continues) and, under n_gpu>1, when an item's featurisation failed or was refused before `predict`
    (`error=featurisation:_…`, stack.item_featurisation_failed); the ITEMS census and the exit code carry the failure."""
    return f"{PREFIX} ITEM event=failed item={_token(rec.get('item') or '?')} N_token={_token(rec.get('N_token'))} error={_token(rec.get('failed'))}"


def items_line(census: dict) -> str:
    """`[protenix-v1-opt] ITEMS total=<n> ok=<k> failed=<f> open=<o> failed_items=<name:Type,...|->` — the per-process item census (stack.items_census)."""
    c = census or {}
    return (f"{PREFIX} ITEMS total={c.get('total', 0)} ok={c.get('ok', 0)} failed={c.get('failed', 0)} open={c.get('open', 0)} "
            f"failed_items={_token(','.join(c.get('failed_items') or []) or '-')}")


def items_exit_code(rc: int, census: dict) -> int:
    """The exit code after the item census: a run whose own code is 0 but with a failed or open item exits EXIT_FAIL (1); a non-zero code stands."""
    c = census or {}
    if rc == EXIT_OK and (c.get("failed", 0) or c.get("open", 0)):
        return EXIT_FAIL
    return rc


def log_items(census: dict, stream=None) -> str:
    return log(items_line(census), stream)


def template_exit_line() -> str:
    """`[protenix-v1-opt] TEMPLATE event=exit templates=off|on [tmpl_items= tmpl_slots= tmpl_real= tmpl_searches= tmpl_hits= tmpl_kept= tmpl_dropped= tmpl_none=]`:
    the template guard's process record (templates.exit_fields), once per process after the LEVER lines."""
    from . import templates as T
    return f"{PREFIX} TEMPLATE event=exit " + " ".join(f"{k}={_token(v)}" for k, v in T.exit_fields().items())


def log_templates(stream=None) -> str:
    return log(template_exit_line(), stream)


def note_line(text: str) -> str:
    """`[protenix-v1-opt] NOTE <text>`: a condition decided up front and NAMED — the run proceeds (never a refusal, never a route change)."""
    return f"{PREFIX} NOTE {text}"


def not_active_line(reason: str) -> str:
    return CORE.not_active_line(TAG, reason)


def partial_detail(partial: List[str], reason: Optional[str]) -> str:
    return PARTIAL_DETAIL_FMT.format(levers=",".join(partial) or "-", reason=reason or "no reason recorded")


def partial_exit_reason(partial: List[str], reason: Optional[str]) -> str:
    """The reason of the partial exit (the text after `NOT ACTIVE: `): `partial activation — <levers>: <reason>; exit 3 (--allow-partial
    records and proceeds)` — what stack passes to ActivationError and every verb prints through `not_active_line`."""
    return PARTIAL_EXIT_REASON_FMT.format(detail=partial_detail(partial, reason))


def partial_exit_line(partial: List[str], reason: Optional[str]) -> str:
    """The partial exit: the NOT ACTIVE line in the family grammar (the one formatter, `not_active_line`)."""
    return not_active_line(partial_exit_reason(partial, reason))


def partial_allowed_line(partial: List[str], reason: Optional[str]) -> str:
    return PARTIAL_ALLOWED_FMT.format(prefix=PREFIX, detail=partial_detail(partial, reason))


def log(line: str, stream=None) -> str:
    return CORE.log_once(None, line, stream)


# ------------------------------------------------------------------------------------------------------------------ the exit rule
def structural_gates() -> Tuple[str, ...]:
    """The count keys of the core's STRUCTURAL refusals — a call the kernel cannot launch at that size by construction, answered by the stock statement by
    design (a named gate on the LEVER line, never partial): the exact TriMul's CUDA grid axis-1 limit (`opt_core.trimul.GRID_REFUSED`, N <= 2047 at C = 128)."""
    from opt_core.trimul import GRID_REFUSED
    return ("stock:unsupported:" + GRID_REFUSED,)


def _split(counts: Dict[str, int], served_keys: Tuple[str, ...], fallback_keys: Tuple[str, ...]) -> Tuple[int, dict, dict]:
    served = sum(v for k, v in counts.items() if k.startswith(served_keys))
    fallback = {k: v for k, v in counts.items() if k.startswith(fallback_keys) and k not in structural_gates() and v}
    gated = {k: v for k, v in counts.items() if k.startswith("stock:") and k not in fallback and v}
    return served, gated, fallback


def floor_state(served: int, min_tokens: Optional[int], items_: Optional[list]) -> Optional[dict]:
    """The flash triangle-attention floor read against the run's items (stack.items(), N_token each) and the floor the package exported
    (report.triattn_floor.exported): {min_tokens, items_atleast, items_below, items_unsized, state} with state `served` (fused calls > 0),
    `floor-off` (no fused call and every sized item below the floor: the stock triangle attention by design) or `partial` (no fused call
    with an item at or above the floor, or an unsized item). None when the floor is unknown."""
    if min_tokens is None:
        return None
    sizes = [it.get("N_token") for it in (items_ or [])]
    atleast = sum(1 for n in sizes if isinstance(n, int) and n >= min_tokens)
    below = sum(1 for n in sizes if isinstance(n, int) and n < min_tokens)
    unsized = len(sizes) - atleast - below
    state = "served" if served else ("floor-off" if below and not atleast and not unsized else "partial")
    return {"min_tokens": int(min_tokens), "items_atleast": atleast, "items_below": below, "items_unsized": unsized, "state": state}


def gate_state(served: int, gate: Optional[dict], items_: Optional[list]) -> Optional[dict]:
    """A size gate of the kit read against the run's items (stack.items(), N_token each): `gate` = {"word", "tokens"} where an item
    with N_token <= tokens is below the gate (the trimul levers: levers_ptx1.TRIMUL_GATE_TOKENS; the transition levers: the largest
    N whose N*N pair rows are under TRANSITION_GATE_ROWS). Returns {word, tokens, items_above, items_atmost, items_unsized, n_max, state}
    with state `served` (lever calls > 0), `gate-off` (no call served and every sized item at or below the gate: the stock statement
    by design, an accounted state, never partial) or `partial` (no call served with an item above the gate, or an unsized item);
    None when the gate is unknown (an older kit account)."""
    if not gate or not isinstance(gate.get("tokens"), int):
        return None
    t = int(gate["tokens"])
    sizes = [it.get("N_token") for it in (items_ or [])]
    above = sum(1 for n in sizes if isinstance(n, int) and n > t)
    atmost = sum(1 for n in sizes if isinstance(n, int) and n <= t)
    unsized = len(sizes) - above - atmost
    n_max = max((n for n in sizes if isinstance(n, int)), default=None)
    state = "served" if served else ("gate-off" if atmost and not above and not unsized else "partial")
    return {"word": gate["word"], "tokens": t, "items_above": above, "items_atmost": atmost, "items_unsized": unsized, "n_max": n_max, "state": state}


def _kit_gates(levers: dict) -> Dict[str, dict]:
    """The kit account's two size gates (levers_ptx1.describe()['gates']) as gate_state inputs per lever family: the trimul levers'
    token gate (`n<=<tokens>`) and the transition levers' row gate expressed as the largest N_token whose N*N rows are below it
    (`rows<<rows>`). {} for an account without them."""
    g = levers.get("gates") if isinstance(levers.get("gates"), dict) else {}
    out: Dict[str, dict] = {}
    if isinstance(g.get("trimul_tokens"), int):
        out["trimul"] = {"word": f"n<={g['trimul_tokens']}", "tokens": int(g["trimul_tokens"])}
    if isinstance(g.get("transition_rows"), int) and g["transition_rows"] > 0:
        import math
        out["transition"] = {"word": f"rows<{g['transition_rows']}", "tokens": math.isqrt(int(g["transition_rows"]) - 1)}   # N*N < rows  <=>  N <= isqrt(rows-1)
    if isinstance(g.get("triattn_gate_tokens"), int) and g["triattn_gate_tokens"] > 0:
        out["triattn"] = {"word": f"n<={g['triattn_gate_tokens']}", "tokens": int(g["triattn_gate_tokens"])}   # the tri-attention levers' structural small-input gate (levers_ptx1.TRIATTN_GATE_TOKENS: upstream's own torch route at or below it)
    if isinstance(g.get("triattn_ceiling_tokens"), int) and g["triattn_ceiling_tokens"] > 0:
        out["triattn_ceiling"] = {"word": f"n>{g['triattn_ceiling_tokens']}", "tokens": int(g["triattn_ceiling_tokens"])}   # the tri-attention levers' CEILING (ceiling_state): above it upstream row-chunks the attention
    return out


def ceiling_state(served: int, ceiling: Optional[dict], items_: Optional[list]) -> Optional[dict]:
    """A size CEILING of the kit read against the run's items: the triangle-attention levers (gblock | gflash and their kernel words) serve
    items of at most `ceiling["tokens"]` (levers_ptx1.TRIATTN_CEILING_TOKENS: above it upstream row-chunks the triangle attention and the
    stock statement answers by name, `stock:chunk`). Returns {word, tokens, items_above, items_atmost, items_unsized, n_min, state} with
    state `served` (lever calls > 0), `ceiling-off` (no call served and every sized item above the ceiling: the stock statement by design,
    an accounted state, never partial — LEVER … state=skipped reason=above_ceiling ceiling=<tokens> n=<n_min>) or `partial`; None when the
    ceiling is unknown (an older kit account)."""
    if not ceiling or not isinstance(ceiling.get("tokens"), int):
        return None
    t = int(ceiling["tokens"])
    sizes = [it.get("N_token") for it in (items_ or [])]
    above = sum(1 for n in sizes if isinstance(n, int) and n > t)
    atmost = sum(1 for n in sizes if isinstance(n, int) and n <= t)
    unsized = len(sizes) - above - atmost
    n_min = min((n for n in sizes if isinstance(n, int)), default=None)
    state = "served" if served else ("ceiling-off" if above and not atmost and not unsized else "partial")
    return {"word": ceiling["word"], "tokens": t, "items_above": above, "items_atmost": atmost, "items_unsized": unsized, "n_min": n_min, "state": state}


def _card_rows(levers: dict, lever: str) -> Optional[dict]:
    """The kit account's row-range facts of `lever` on this card (`card_rows`: levers_ptx1.row_range_facts), None when it has none."""
    cr = levers.get("card_rows") if isinstance(levers.get("card_rows"), dict) else {}
    r = cr.get(lever)
    return r if isinstance(r, dict) and "min_rows" in r else None


def range_state(served: int, counts: Dict[str, int], rows: Optional[dict]) -> Optional[dict]:
    """An exact-class pair lever's served row range on this card (levers_ptx1.row_range_facts: {cc, min_rows, max_rows, below, above}) read
    against the lever's census: {min_rows, max_rows, calls_below, calls_above, state} with state `served` (lever calls > 0), `range-off` (no
    lever call and calls outside the range: the stock forward by design at this input size) or `partial` (neither). None when the card has
    no range for the lever (nothing printed)."""
    if not rows or "min_rows" not in rows:
        return None
    below = int(counts.get(rows.get("below") or "", 0) or 0); above = int(counts.get(rows.get("above") or "", 0) or 0)
    state = "served" if served else ("range-off" if (below or above) else "partial")
    return {"min_rows": int(rows["min_rows"]), "max_rows": int(rows["max_rows"]), "calls_below": below, "calls_above": above, "state": state}


CORE_HOSTS = {"tricuda": ("fast", "gflash"), "triexact": ("exact", "gblock")}    # the provider words: (the tier word each binds by default, the lever each rides) — levers_ptx1.CORE_WORDS / core_tier


def core_evidence(lever: str, acct: Optional[dict]) -> dict:
    """The evidence of a provider word (`tricuda` | `triexact`) from the kit account (levers_ptx1.describe(): `cfg`, and `counts` — the families
    `core:<word>` = the census {<row>: n served through the provider (whatever row the TIER word named per key, as the provider's cells
    name it on this card), `cueq:<refusal>`: n answered by the stock statement BY NAME (a key the tier word refused: another card, dtype,
    no cell, no prebuilt ABI; the library's torch-path threshold; a row's own per-call words), `error:<Type>`: n (an exception on a call: read as
    partial)}, `coresel:<word>` = one `<key>=><selection | refused:<kind>-><row>>` entry per selection key, `corefact:<word>` = `tier=<word>` /
    `form=` / `prebuilt=<stack key>:built|missing` / `exact_stack=<vouch key>`, `coremember:<word>` = the exact member's own census {served, calls,
    refused:<reason>} -> `member=<served>/<calls>` [`member_refused=<reason>:n,…`]). served / gated / fallback as above; `row` = the row that
    served the most calls (`none` when nothing served), `served_by` = every row's count when more than one served, `tier` = the tier word bound;
    `aside` = the word's by-design state when it served nothing and nothing failed: the lever it rides is not on this arm (`host_absent:<lever>`),
    every call was refused by name (the top refusal word), or the host made no core call at all (`no_core_call`: outside the row range, gated)
    — accounted, never partial."""
    tier, host = CORE_HOSTS[lever]
    acct = acct if isinstance(acct, dict) else {}
    counts = acct.get("counts") if isinstance(acct.get("counts"), dict) else {}
    cfg = acct.get("cfg") if isinstance(acct.get("cfg"), dict) else {}
    c = counts.get("core:" + lever) if isinstance(counts.get("core:" + lever), dict) else {}
    by = {k: int(v) for k, v in c.items() if ":" not in k and isinstance(v, int) and not isinstance(v, bool) and v}   # bare tokens = provider rows that served
    served = int(sum(by.values()))
    fallback = {k: v for k, v in c.items() if k.startswith("error:") and v}
    gated = {k: v for k, v in c.items() if ":" in k and not k.startswith("error:") and v}
    sels = sorted((counts.get("coresel:" + lever) or {}).keys()) if isinstance(counts.get("coresel:" + lever), dict) else []
    facts = sorted((counts.get("corefact:" + lever) or {}).keys()) if isinstance(counts.get("corefact:" + lever), dict) else []
    for fct in facts:                                                              # the tier word the kit bound this process (levers_ptx1.core_tier), when the account says
        k, _, v = str(fct).partition("=")
        if k == "tier" and v:
            tier = v
    facts = [f for f in facts if not str(f).startswith("tier=")]
    mem = counts.get("coremember:" + lever) if isinstance(counts.get("coremember:" + lever), dict) else {}
    if mem:                                                                        # the exact member behind the word: calls its kernel served / calls handed to it, its by-name refusals (the stock statement served those)
        facts.append("member=%d/%d" % (int(mem.get("served") or 0), int(mem.get("calls") or 0)))
        ref = {str(k).split(":", 1)[1]: int(v) for k, v in mem.items() if str(k).startswith("refused:") and v}
        if ref:
            facts.append("member_refused=" + ",".join("%s:%d" % kv for kv in sorted(ref.items(), key=lambda kv: (-kv[1], kv[0]))))
    row = sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if by else "none"
    e = {"served": served, "gated": gated, "fallback": fallback, "retried": {}, "row": row, "tier": tier, "host": host, "selections": sels, "facts": ["tier=%s" % tier] + facts}
    if len(by) > 1:                                                                # more than one row served (the cell names rows per row-length bucket): each row's count
        e["served_by"] = dict(sorted(by.items()))
    if not served and not fallback:
        if host in cfg and not cfg.get(host):
            word = "host_absent:%s" % host
        elif gated:
            word = sorted(gated.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0][0]
        else:
            word = "no_core_call"
        e["aside"] = {"state": "aside", "word": word}
    return e


def kit_evidence(levers: Optional[dict], trimul: str, mode_levers, items_: Optional[list] = None, triattn_floor: Optional[int] = None) -> Dict[str, dict]:
    """The kit's own account (`levers_ptx1.describe()`, JSON-safe: stack.kit_lever_summary) read lever by lever for the mode's arm:
    {lever: {"served": n, "gated": {counter: n}, "fallback": {counter: n}, "retried": {counter: n}}}. served = the calls the lever
    answered; gated = the kit's own shape gates (the stock path by design: `stock:*`, `eager_gate`, `bypass`); fallback = the stock path
    taken AFTER the lever was engaged. The counter names are the kit's (`opt/forward/v05_addon`):
      trimul <exact|fast>  ptxfpf/levers_ptx1.py:91/:78 `exact`/`fast` served; :93 `error:<Exc>` = kernel error -> stock for this call
                           (fallback); `stock:unsupported:<word>` = the core refused the cell -> stock for that call: the structural word
                           (`structural_gates()`: the exact TriMul's launch-grid limit) is a gate, any other word a fallback;
                           :61/:65 `stock:path` / `stock:N` gates; `buf_release` = stack.release_trimul_buffers' record (the exact
                           TriMul's persistent per-(N, dtype) scratch buffers, opt_core/kernels/fpf_trimul/kernels.py `_BUF`, dropped at every
                           item entry so one item's set is resident, never every distinct N of the process: state on|off:<why>|na, released, calls).
      gblock | gflash      levers_ptx1.py `_tri_forward`: `gblock` / `gflash` / `gflash:cueq<floor` served (the last: the block around the
                           cuEquivariance core for an item under the flash floor); `stock:path` / `stock:chunk` / `stock:gate` gates;
                           `stock:below_min_rows` / `stock:above_max_rows` = a gblock call outside the block's served row range on this
                           card (opt_core.attn.pair_fused exact_rows; the account's `card_rows` names the range) -> the stock forward (gated);
                           `fallback:<lever>:<reason>` = the core refused the call before any launch (opt_core.attn.pair_fused.Unsupported:
                           a cell, dtype or shape outside the core's table) -> stock for that call (fallback); `degraded.fast_launch` = flash_triattn's
                           direct launcher disabled itself (`flash_triattn_launch`: fast_launch_stats() `disabled` / `why_disabled`) ->
                           plain Triton JIT launches, identical numerics (named, not partial).
      xtr | ttr            levers_ptx1.py `_transition_forward`: `xtr:C=<C>` / `ttr:C=<C>` served; `stock:C=<C>` gates (widths / dtypes the levers do not own);
                           `stock:below_min_rows` / `stock:above_max_rows` = an xtr call outside its served rows on this card (opt_core.attn.pair_fused exact_rows) -> stock (gated); `stock:<lever>:<refusal>` / `stock:<lever>:row=<stock row>` = the provider refused the call by name / its tier is the stock statement for this cell (gated);
                           `fallback:<lever>:<reason>` = the core refused the call (fallback).
      sg                   lib/kit112_src/infopt_graphs/protenix/graphed.py:325 `replays` served (:495 `captures`); :228/:329
                           `eager_steps` = the stock step (fallback); :237 `bypass` gate (above the loop's max_tokens);
                           sampler `graphs` False = not installed (fallback).
      hoist                lib/kit112_src/dit_hoist.py:136 `hits` + :144 `records` served; :130 `bypass` gate; `hoist_installed`
                           False = not installed (fallback).
    A kit account that is absent or carries `error` gives every lever of the mode fallback {"account": <error>}."""
    out: Dict[str, dict] = {}
    has_trimul = bool(trimul) and trimul != "stock"          # `stock` = no triangle-multiplication lever on the arm (MODEL_OPT_LEVERS_OFF ablated the word): the other levers are still read
    want = ([trimul] if has_trimul else []) + [lv for lv in (mode_levers or ()) if lv in ("gblock", "gflash", "tricuda", "triexact", "xtr", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "dit_attn_exact", "lazy_init", "template_dedupe", "tmpl_triatt", "tmpl_trimul", "tmpl_xtr", "tmpl_pairfused", "sampler_prep", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact", "tmpl_trimul_exact")]
    if not levers or levers.get("error") or not isinstance(levers, dict):
        why = (levers or {}).get("error") if isinstance(levers, dict) else None
        for lv in want:
            out[lv] = {"served": 0, "gated": {}, "fallback": {"account": why or "the kit's account is missing (levers_ptx1.describe() was never read)"}, "retried": {}}
        return out
    counts = levers.get("counts") or {}
    smp = levers.get("sampler") if isinstance(levers.get("sampler"), dict) else {}
    gates = _kit_gates(levers)
    c = counts.get("trimul") or {}
    served, gated, fallback = _split(c, (trimul,), ("error:", "stock:unsupported"))
    if has_trimul:
        out[trimul] = {"served": served, "gated": gated, "fallback": fallback, "retried": {}}
        selx = counts.get("coresel:trimul_exact") if trimul == "exact" else None                 # the exact TriMul through the provider (row tmk3_exact): its selections per key
        if isinstance(selx, dict) and selx:
            out[trimul]["row"] = "tmk3_exact"; out[trimul]["selections"] = sorted(selx)
        if trimul in ("fast", "exact"):                                            # the TriMul words through the provider (levers_ptx1._provider_trimul; _exact_provider_trimul):
            cf = counts.get("core:trimul_" + trimul) if isinstance(counts.get("core:trimul_" + trimul), dict) else {}   # COUNTS["core:trimul_<word>"] = {<row served>: n, `next:<word>:<kind>`: n handed on by name,
            self_ = counts.get("coresel:trimul_" + trimul) if isinstance(counts.get("coresel:trimul_" + trimul), dict) else {}   # `error:<word>:<Type>`: n}; COUNTS["coresel:trimul_<word>"] = one selection note per (word, key)
            floor = "kit" if trimul == "fast" else "tmk3_exact"                       # the word's own statement: the calls the provider did not serve (fast: the generic face; exact: the tmk3_exact kernels themselves)
            by = {k: int(v) for k, v in cf.items() if isinstance(v, int) and v and not k.startswith(("next:", "error:"))}
            kit_n = max(0, int(served) - sum(by.values()))
            if kit_n and (by or cf):
                by[floor] = by.get(floor, 0) + kit_n
            if by:
                out[trimul]["row"] = sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                if len(by) > 1:
                    out[trimul]["served_by"] = dict(sorted(by.items()))
            elif cf or self_:
                out[trimul]["row"] = floor
            nxt = {k: int(v) for k, v in cf.items() if k.startswith("next:") and v}
            err = {k: int(v) for k, v in cf.items() if k.startswith("error:") and v}
            if nxt:
                out[trimul]["gated"] = dict(out[trimul].get("gated") or {}, **nxt)
            if err:                                                                # a row's exception (the next word answered the call): read as partial, like the statement's own errors
                out[trimul]["fallback"] = dict(out[trimul].get("fallback") or {}, **err)
            if self_:
                out[trimul]["selections"] = sorted(self_)
            tf = levers.get("trimul_" + trimul) if isinstance(levers.get("trimul_" + trimul), dict) else {}
            facts = []
            if tf.get("env_word"):
                facts.append("word_env=%s" % tf["env_word"])
            if tf.get("bind"):
                facts.append("bind=%s" % tf["bind"])
            if facts:
                out[trimul]["facts"] = facts
        byname = {k: v for k, v in (out[trimul].get("gated") or {}).items() if str(k).startswith("stock:trimul:") and v}   # the provider refused the key BY NAME
        if not served and byname and not out[trimul].get("fallback"):             # (no row for this card / dtype / width at the tier word) for every call: upstream's statement by design,
            out[trimul]["aside"] = {"state": "aside", "word": sorted(byname.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0][0]}   # accounted `state=skipped reason=aside`, exit 0 — never partial
        gs = gate_state(served, gates.get("trimul"), items_)                        # the trimul token gate read against the items (gate-off: every item at or below it)
        if gs is not None:
            out[trimul]["gate"] = gs
    if has_trimul and isinstance(levers.get("trimul_buffers"), dict):              # stack.release_trimul_buffers: the exact TriMul's persistent
        out[trimul]["buf_release"] = dict(levers["trimul_buffers"])                #   per-N buffers released at every item entry (state, released, calls)
    if "gflash" in want or "gblock" in want:                                       # gflash wins over gblock when both are named (levers_ptx1._tri_forward)
        lv = "gflash" if "gflash" in want else "gblock"
        c = counts.get("triattn") or {}
        served, gated, fallback = _split(c, (lv,), ("fallback:",))                # `gflash` also counts `gflash:cueq<floor` (the block served the call around the cuEquivariance core)
        out[lv] = {"served": served, "gated": gated, "fallback": fallback, "retried": {k: v for k, v in c.items() if k.startswith("kernel-error") and v}}
        rs = range_state(served, c, _card_rows(levers, lv))                         # gblock's served row range on this card, when it has one (gflash never has)
        if rs is not None:
            out[lv]["rows"] = rs
        cs = ceiling_state(served, gates.get("triattn_ceiling"), items_)           # the tri-attention token ceiling read against the items (ceiling-off: every item above it, `stock:chunk`)
        if cs is not None:
            out[lv]["ceiling"] = cs
        gs = gate_state(served, gates.get("triattn"), items_)                       # upstream's small-input route read against the items (gate-off: every item at or below it, `stock:gate`)
        if gs is not None:
            out[lv]["gate"] = gs
        if lv == "gflash":
            out[lv]["floor"] = floor_state(int(c.get("gflash") or 0), triattn_floor, items_)   # a caller's floor, when the account carries one (None: the kit has no tri-attention size floor)
            gc = counts.get("gflash_core") if isinstance(counts.get("gflash_core"), dict) else {}   # levers_ptx1.gflash_core: the block's core per call — `tricuda:<row>` (the provider row the
            if gc:                                                                 # tier word named), `cueq:<refusal>` (a key the tier word refused: the stock statement), `default` (the word off) -> `cores=<word>:n,…`
                out[lv]["facts"] = ["cores=%s" % ",".join("%s:%d" % (k, int(v)) for k, v in sorted(gc.items(), key=lambda kv: (-int(kv[1]), kv[0])))]
            fl = levers.get("flash_triattn_launch") if isinstance(levers.get("flash_triattn_launch"), dict) else {}
            if fl.get("disabled"):                                                 # opt_core/kernels/flash_triattn.py fast_launch_stats(): the direct launcher is off -> plain JIT launches (identical numerics, slower)
                out[lv]["degraded"] = {"fast_launch": {"state": "disabled", "why": fl.get("why_disabled"), "fast_launches": int(fl.get("fast_launches") or 0), "jit_launches": int(fl.get("jit_launches") or 0)}}
    for lv in ("triexact", "tricuda"):                                            # ride gblock / gflash: the block's attention core through opt_core.kernels.triattn by tier word (core_evidence reads counts core:/coresel:/corefact:/coremember: + cfg)
        if lv not in want:
            continue
        out[lv] = core_evidence(lv, levers)
    if "xtr" in want or "ttr" in want:                                             # xtr wins over ttr when both are named (levers_ptx1._transition_forward)
        lv = "xtr" if "xtr" in want else "ttr"
        c = counts.get("transition") or {}
        served, gated, fallback = _split(c, (lv + ":",), ("fallback:",))
        out[lv] = {"served": served, "gated": gated, "fallback": fallback, "retried": {}}
        tv = levers.get("transition") if isinstance(levers.get("transition"), dict) else {}   # the provider account (levers_ptx1.describe()["transition"]): the tier word asked
        if tv.get("word"):                                                          # (`word=exact|fast|big`) and the Selection line served per (lever, family, c, hidden) (`selections=`)
            out[lv]["facts"] = ["word=%s" % tv["word"]]
        if tv.get("selections"):
            out[lv]["selections"] = sorted((tv.get("selections") or {}).values())
        rs = range_state(served, c, _card_rows(levers, lv))                         # xtr's served rows on this card, when the core's rule has an entry for it (ttr never has)
        if rs is not None:
            out[lv]["rows"] = rs
        gs = gate_state(served, gates.get("transition"), items_)                    # the transition row gate read against the items
        if gs is not None:
            out[lv]["gate"] = gs
        byname = {k: v for k, v in gated.items() if str(k).startswith("stock:%s:row=" % lv) and v}   # as the TriMul rule above: the provider ANSWERED the eligible calls and its
        if not served and byname and not fallback:                                  # answer for every one was a STOCK row BY NAME (`stock:<lever>:row=torch_swiglu` — `exact` at 9.0|bf16|pair_c128_n4|N<=256,
            out[lv]["aside"] = {"state": "aside", "word": sorted(byname.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0][0]}   # where the table names no exact-class kernel row): the module's own forward IS the
                                                                                     # table's answer at this size — accounted `LEVER … state=skipped reason=aside aside=<word>`, exit 0, never partial (a stock-row
                                                                                     # tier answers with the stock module by name). A provider error after engagement
                                                                                     # (`fallback:*`) and a lever that met NO eligible call (only `stock:C=<C>` dtype/width gates) stay partial.
    if "sg" in want:
        s = smp.get("sampler") if isinstance(smp.get("sampler"), dict) else {}
        fallback = {k: s[k] for k in ("eager_steps",) if s.get(k)}
        if not smp.get("graphs"):
            fallback["graphs"] = "not installed"
        out["sg"] = {"served": int(s.get("replays") or 0), "gated": {k: s[k] for k in ("bypass",) if s.get(k)}, "fallback": fallback, "retried": {}}
    if "hoist" in want:
        h = smp.get("hoist") if isinstance(smp.get("hoist"), dict) else {}
        fallback = {} if smp.get("hoist_installed") else {"hoist_installed": "not installed"}
        out["hoist"] = {"served": int(h.get("hits") or 0) + int(h.get("records") or 0), "gated": {k: h[k] for k in ("bypass",) if h.get(k)}, "fallback": fallback, "retried": {}}
    if "keep_pool" in want:                                                         # lib/ptx1_keep_pool.report(): the stock in-forward empty_cache calls skipped (served) and the
        kp = levers.get("keep_pool") if isinstance(levers.get("keep_pool"), dict) else {}   # releases passed through (the per-item runner site, the kit's own) counted apart; not installed = fallback
        fallback = {} if kp.get("installed") else {"keep_pool_installed": kp.get("error") or "not installed"}
        if int(kp.get("errors") or 0):
            fallback = dict(fallback, bookkeeping_errors=int(kp["errors"]))
        out["keep_pool"] = {"served": int(kp.get("skipped_total") or 0), "gated": ({"passed": int(kp["passed_total"])} if kp.get("passed_total") else {}), "fallback": fallback, "retried": {}}
    if "summary_hostidx" in want:                                                   # lib/ptx1_summary_host.report(): served = samples summarised on the host-index path; delegated
        sh = levers.get("summary_hostidx") if isinstance(levers.get("summary_hostidx"), dict) else {}   # (non-inference call shapes -> stock) and stepped_aside (the row-sharded line owns
        gated = {k: int(sh[k]) for k in ("delegated",) if sh.get(k)}               # the function: named on the BIG line, excused there) are gated by design; not installed
        if sh.get("stepped_aside"):                                                 # otherwise / bookkeeping errors = fallback
            gated["stepped_aside"] = 1
        fallback = {} if (sh.get("installed") or sh.get("stepped_aside")) else {"summary_hostidx_installed": sh.get("error") or "not installed"}
        if int(sh.get("errors") or 0):
            fallback = dict(fallback, errors=int(sh["errors"]))
        out["summary_hostidx"] = {"served": int(sh.get("samples") or 0), "gated": gated, "fallback": fallback, "retried": {}}
    apb = levers.get("apb") if isinstance(levers.get("apb"), dict) else {}           # apb_ptx1.describe(): {"dit": facts, "atom": facts}
    for lv in ("ditattn", "ditattnfp16", "atomattn"):
        if lv not in want:
            continue
        f = apb.get("atom" if lv == "atomattn" else "dit") or {}
        c = counts.get("atom" if lv == "atomattn" else "dit") or {}
        served, gated, fallback = _split(c, ("apb:",), ("error:",))
        gated = dict(gated, **{k: v for k, v in c.items() if k.startswith("named:") and v})
        if lv == "ditattnfp16" and not f.get("fp16"):
            served = 0
        if not f.get("engaged"):
            fallback = dict(fallback, engaged="not installed: %s" % (f.get("named") or "not requested"))
        out[lv] = {"served": served, "gated": gated, "fallback": fallback, "retried": {}, "cell": "%s@%s" % (f.get("opd"), f.get("cell_key")),
                   "installed_on": f.get("installed_on"), "calls": f.get("calls"), **({"named": f["named"]} if f.get("named") else {})}
    if "dit_attn_exact" in want:                                                    # opt_core/kernels/apb/dit_exact report(): served = calls routed to the kernel; calls that took the
        dx = levers.get("dit_attn_exact") if isinstance(levers.get("dit_attn_exact"), dict) else {}   # stock statement by envelope rule (rank / head_dim / dtype / no_bias …) are gated by reason; a card
        routes = dict(dx.get("routes") or {})                                       # or stack without a prebuilt is `card-off` (the lever steps aside by name: report.partial_of, exit 0);
        served = int(routes.pop("kernel", 0) or 0)                                  # not installed for any other reason (digest / load-check) = fallback
        gated = {k: int(v) for k, v in routes.items() if v}
        e = {"served": served, "gated": gated, "fallback": {}, "retried": {}}
        if dx.get("card_off"):
            e["card"] = {"state": "card-off", "why": str(dx["card_off"])}
        elif not dx.get("installed"):
            e["fallback"] = {"dit_attn_exact_installed": dx.get("error") or "not installed"}
        out["dit_attn_exact"] = e
    if "lazy_init" in want:                                                         # served = model constructions that ran with the random init skipped (1 per process); patched = init
        lz = levers.get("lazy_init") if isinstance(levers.get("lazy_init"), dict) else {}     # functions no-op'd; recheck (PTX_LAZY_INIT=recheck) = the N/N state-tensor comparison
        e = {"served": int(lz.get("constructs") or 0), "gated": {}, "fallback": {}, "retried": {}, "construct_s": lz.get("lazy_construct_s"), "patched": lz.get("patched")}
        rc = lz.get("recheck") if isinstance(lz.get("recheck"), dict) else None
        if rc and rc.get("verdict"):
            e["recheck"] = f"{rc.get('verdict')}:{rc.get('n_equal')}/{rc.get('n_tensors')}"
            if rc.get("verdict") != "IDENTICAL":                                    # the loaded model differs from the stock-initialised one: a failed lever, never a silent pass
                e["fallback"] = {"recheck": e["recheck"]}
        if not lz.get("installed"):
            e["fallback"] = dict(e["fallback"], lazy_init_installed=lz.get("error") or "not installed")
        out["lazy_init"] = e
    if "sampler_prep" in want:                                                      # served = the graphed sampler's captures + replays that ran through the host path (all six parts on);
        smp = levers.get("sampler") if isinstance(levers.get("sampler"), dict) else {}         # facts: parts word, poison verdict, pool_chained / pool_renewed / key_value_miss; a partial parts word
        pr = smp.get("prep") if isinstance(smp.get("prep"), dict) else {}                       # or a failed poison probe = fallback by name; without the sampler graph (sg ablated) it stepped
        st_ = pr.get("stats") if isinstance(pr.get("stats"), dict) else {}                     # aside by name: rows range-off no_sampler_graph (exit 0)
        sm = smp.get("sampler") if isinstance(smp.get("sampler"), dict) else {}
        served = (int(sm.get("captures") or 0) + int(sm.get("replays") or 0)) if pr.get("on") else 0
        gated = {k: int(st_[k]) for k in ("key_value_miss", "pool_chained", "pool_renewed", "poison_skipped") if st_.get(k)}
        e = {"served": served, "gated": gated, "fallback": {}, "retried": {}, "parts": pr.get("parts") or "-", "poison": pr.get("poison") or "-"}
        if pr.get("aside"):
            e["rows"] = {"state": "range-off", "why": str(pr["aside"])}
        elif not pr.get("on"):
            e["fallback"] = {"sampler_prep": "not installed"}
        elif pr.get("parts") != "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec":
            e["fallback"] = {"parts": pr.get("parts")}
        elif str(pr.get("poison") or "").startswith("failed"):
            e["fallback"] = {"poison": pr.get("poison")}
        out["sampler_prep"] = e
    tp_ = levers.get("templ") if isinstance(levers.get("templ"), dict) else {}          # ptxfpf/ptx1_templ.describe(): the template-embedder levers' accounts
    if "template_dedupe" in want:                                                   # served = template evaluations run through the lever (evaluated); reused / slots / groups are facts;
        td = tp_.get("template_dedupe") if isinstance(tp_.get("template_dedupe"), dict) else {}   # not installed = fallback; the stock words (untemplated / unkeyed) are gated counts
        fallback = {} if td.get("installed") else {"template_dedupe_installed": td.get("error") or "not installed"}
        out["template_dedupe"] = {"served": int(td.get("evaluated") or 0), "gated": {**{("stock_" + k): v for k, v in (td.get("stock") or {}).items()}, **({"reused": int(td["reused"])} if td.get("reused") else {})}, "fallback": fallback, "retried": {}}
        if td.get("installed") and not int(td.get("calls") or 0):                  # installed, never called: the class method is not on this run's path (n_gpu > 1: tp.template_tp transcribes the
            out["template_dedupe"]["rows"] = {"state": "range-off", "why": "not_on_path"}   # embedder on rows; a checkpoint with n_blocks = 0) — nothing to serve, by design (exit 0), never a partial activation
    if "tmpl_triatt" in want:                                                       # served = template triangle-attention calls routed through the block; the by-name stock words are gated counts
        tt = tp_.get("tmpl_triatt") if isinstance(tp_.get("tmpl_triatt"), dict) else {}     # (a card without the cells: every call `stock:tmpl_triatt:<word>` = range-off by design, exit 0)
        gated = {("stock_" + k.replace(":", "_")): v for k, v in (tt.get("stock") or {}).items()}
        out["tmpl_triatt"] = {"served": int(tt.get("routed_total") or 0), "gated": gated, "fallback": {} if tt.get("on") else {"tmpl_triatt": "not set"}, "retried": {}}
        if not out["tmpl_triatt"]["served"] and tt.get("on"):                       # on, nothing routed: no servable cell on this card (the stock words say which) or the template attention is
            out["tmpl_triatt"]["rows"] = {"state": "range-off", "why": ",".join(sorted(gated)) or "not_on_path"}   # not on this run's path (n_gpu > 1) — by design, exit 0
    if "tmpl_trimul" in want:                                                       # served = template TriMul calls the provider answered; stock words (a refusal `<row>:<kind>`, small items)
        tm = tp_.get("tmpl_trimul") if isinstance(tp_.get("tmpl_trimul"), dict) else {}     # are gated counts; an import failure while on = fallback
        gated = {("stock_" + k.replace(":", "_")): v for k, v in (tm.get("stock") or {}).items()}
        fallback = {"tmpl_trimul": tm.get("error")} if tm.get("error") else ({} if tm.get("on") else {"tmpl_trimul": "not set"})
        out["tmpl_trimul"] = {"served": int(tm.get("routed_total") or 0), "gated": gated, "fallback": fallback, "retried": {}}
        if not out["tmpl_trimul"]["served"] and tm.get("on") and not tm.get("error"):   # on, nothing routed: every call refused or gated by name, or the template TriMul is not on this run's path
            out["tmpl_trimul"]["rows"] = {"state": "range-off", "why": ",".join(sorted(gated)) or "not_on_path"}   # (n_gpu > 1) — by design, exit 0
    if "tmpl_trimul_exact" in want:                                                       # served = template TriMul calls the provider answered; stock words (a refusal `<row>:<kind>`, small items)
        tm = tp_.get("tmpl_trimul_exact") if isinstance(tp_.get("tmpl_trimul_exact"), dict) else {}     # are gated counts; an import failure while on = fallback
        gated = {("stock_" + k.replace(":", "_")): v for k, v in (tm.get("stock") or {}).items()}
        fallback = {"tmpl_trimul_exact": tm.get("error")} if tm.get("error") else ({} if tm.get("on") else {"tmpl_trimul_exact": "not set"})
        out["tmpl_trimul_exact"] = {"served": int(tm.get("routed_total") or 0), "gated": gated, "fallback": fallback, "retried": {}}
        if not out["tmpl_trimul_exact"]["served"] and tm.get("on") and not tm.get("error"):   # on, nothing routed: every call refused or gated by name, or the template TriMul is not on this run's path
            out["tmpl_trimul_exact"]["rows"] = {"state": "range-off", "why": ",".join(sorted(gated)) or "not_on_path"}   # (n_gpu > 1) — by design, exit 0
    if "tmpl_xtr" in want:                                                          # served = template transition calls in the exact construction; an unserved cell
        tx = tp_.get("tmpl_xtr") if isinstance(tp_.get("tmpl_xtr"), dict) else {}           # (stock words) is gated by name; a refusal at call time = fallback
        gated = {("stock_" + k.replace(":", "_")): v for k, v in (tx.get("stock") or {}).items()}
        fb = {("fallback_" + k.replace(":", "_")): v for k, v in (tx.get("fallback") or {}).items()}
        if not tx.get("on"): fb = {"tmpl_xtr": "not set"}
        out["tmpl_xtr"] = {"served": int(tx.get("routed_total") or 0), "gated": gated, "fallback": fb, "retried": {}}
        if not out["tmpl_xtr"]["served"] and tx.get("on") and not fb:                # on, nothing routed: no cell on this card (by name) or not on this run's path — by design, exit 0
            out["tmpl_xtr"]["rows"] = {"state": "range-off", "why": ",".join(sorted(gated)) or "not_on_path"}
    if "tmpl_pairfused" in want:                                                    # served = template block calls that took the fpf prologue / epilogue; nothing to serve
        tf = tp_.get("tmpl_pairfused") if isinstance(tp_.get("tmpl_pairfused"), dict) else {}   # without tmpl_triatt's routed calls (range-off by design)
        out["tmpl_pairfused"] = {"served": int(tf.get("calls") or 0), "gated": {}, "fallback": {} if tf.get("on") else {"tmpl_pairfused": "not set"}, "retried": {}}
        if not out["tmpl_pairfused"]["served"] and tf.get("on"):
            out["tmpl_pairfused"]["rows"] = {"state": "range-off", "why": "no_routed_template_calls"}
    t2 = levers.get("trunk2") if isinstance(levers.get("trunk2"), dict) else {}      # trunk2_ptx1.describe(): {"pf": facts, "opm": facts, "pwa": facts}; served = calls the fused
    for lv, kind in (("pfattn", "pf"), ("opm_fused", "opm"), ("pwa_fused", "pwa")):    # kernels answered (`t2:<kind>@<cell>`), gated = calls the module's own statement answered by
        if lv not in want:                                                          # envelope rule (`stock:<reason>`) + modules not installed on by name (`named:`), fallback = kernel
            continue                                                                # errors (`error:<Type>`); installed and never entered this run (no MSA pass) = aside, by design
        f = t2.get(kind) or {}
        c = counts.get(kind) or {}
        served, gated, fallback = _split(c, ("t2:",), ("error:",))
        gated = dict(gated, **{k: v for k, v in c.items() if k.startswith("named:") and v})
        if not f.get("engaged"):
            fallback = dict(fallback, engaged="not installed: %s" % (f.get("named") or "not requested"))
        e = {"served": served, "gated": gated, "fallback": fallback, "retried": {}, "cell": "%s@%s" % (kind, f.get("cell_key")),
             "installed_on": f.get("installed_on"), "calls": f.get("calls"), **({"named": f["named"]} if f.get("named") else {})}
        if f.get("engaged") and not served and not fallback and not any(str(k).startswith("stock:") for k in gated):
            e["aside"] = {"state": "aside", "word": "no_call"}                        # the model never entered the site this run (an input without MSA features returns before the blocks)
        out[lv] = e
    df = levers.get("ditfast") if isinstance(levers.get("ditfast"), dict) else {}         # ditfast_ptx1.describe(): {"words": {word: account}} — the fused-sampler levers
    dfw = df.get("words") if isinstance(df.get("words"), dict) else {}                  # (cond_dedupe / dit_fused / dit_lowp / atom_fused / atom_attn_exact): served = the calls the
    for lv in ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact"):  # lever answered (Python-level: inside the sampler graph the record + capture steps), gated = the
        if lv not in want:                                                              # stock statement by the lever's own rule (cond_dedupe's non-expanded t_hat, atom_attn_exact's
            continue                                                                    # envelope misses), fallback = an install error or a dependency the caller withheld (dit_fused
        a = dfw.get(lv) if isinstance(dfw.get(lv), dict) else {}                       # without ditattn, dit_lowp without dit_fused, atom_fused without atomattn: a partial arm),
        e = {"served": int(a.get("served") or 0), "gated": dict(a.get("gated") or {}), "fallback": dict(a.get("fallback") or {}), "retried": {},
             "wfacts": {k2: v2 for k2, v2 in (a.get("facts") or {}).items() if v2 is not None}}   # card = no cell on this card (steps aside by name, exit 0), aside = the package refused by
        if not a:                                                                       # name (routes / digest / slot owned by atomattn): the stock statement by design, exit 0
            e["fallback"] = {"account_missing": 1}
        elif a.get("card"):
            e["card"] = {"state": "card-off", "why": str(a["card"])}
        elif a.get("aside"):
            e["aside"] = {"state": "aside", "word": _token(str(a["aside"]))[:120]}
        out[lv] = e
    for fused, base_words in (("dit_fused", ("ditattn", "ditattnfp16")), ("atom_fused", ("atomattn",))):   # census interaction (`subsumes`): while a fused
        a = dfw.get(fused) if isinstance(dfw.get(fused), dict) else {}                  # stack is engaged its attention IS the base lever's kernel (protenix_fpf_apb.dit_apb / atom_apb) —
        if not (a and a.get("state") == "on"):                                          # the base lever's own per-call wrapper is not entered, so its census is re-based on the stack's
            continue                                                                    # calls (`subsumed_by=<fused>`), never read as served 0
        for b in base_words:
            eb = out.get(b)
            if eb is None or eb["fallback"] or eb["served"]:
                continue
            if b == "ditattnfp16" and not ((apb.get("dit") or {}).get("fp16")):
                continue
            eb["served"] = int(a.get("served") or 0); eb["subsumed_by"] = fused
    return out


STOCK_KNOB_DEFAULTS = {"triatt_kernel": "cuequivariance", "trimul_kernel": "cuequivariance", "dtype": "bf16"}   # upstream's kernel / dtype knobs (configs.triangle_attention,
                                                                                                                  # .triangle_multiplicative, .dtype) at the values the kit's kernels serve
KNOB_LEVERS = {                                                    # the levers each knob routes off the kernel path when set to another value the upstream accepts
    "triatt_kernel": ("gblock", "gflash", "tricuda", "triexact", "tmpl_triatt", "tmpl_pairfused"),                # --triatt_kernel torch|triattention|deepspeed: the
                                                                   # stock triangle attention answers every call (stock:path), the fused block and its kernel words idle
    "trimul_kernel": ("exact", "fast", "tmpl_trimul"),             # --trimul_kernel torch: the stock TriMul statement answers every call (stock:path)
    "dtype": ("gblock", "gflash", "tricuda", "triexact", "xtr", "ttr", "pfattn", "opm_fused", "pwa_fused", "exact", "fast",     # --dtype fp32|fp16: autocast off or
              "tmpl_triatt", "tmpl_pairfused", "tmpl_trimul", "tmpl_xtr", "ditattn", "ditattnfp16", "atomattn",                          # another operand dtype — the bf16 /
              "dit_attn_exact", "atom_attn_exact", "dit_fused", "dit_lowp", "atom_fused", "cond_dedupe"),                                # autocast-gated levers idle (stock:gate,
}                                                                  # stock:no_autocast, stock:dtype=…); a lever of the set that still serves under the knob is judged as always


def stock_knobs(rep: Optional[dict]) -> Dict[str, str]:
    """The run's stock kernel / dtype knobs as the runner wrap read them off upstream's configs (stack: rep['stock_knobs']), reduced to
    the ones set AWAY from the value the kit's kernels serve: {knob: value}. {} on default knobs or an older record."""
    k = (rep or {}).get("stock_knobs") if isinstance((rep or {}).get("stock_knobs"), dict) else {}
    return {n: str(k[n]) for n in STOCK_KNOB_DEFAULTS if n in k and k[n] is not None and str(k[n]) != STOCK_KNOB_DEFAULTS[n]}


def knob_states(evidence: Dict[str, dict], rep: Optional[dict]) -> Dict[str, dict]:
    """Mark, in place, every lever of the mode that a user-chosen STOCK knob routed off the kernel path: a knob of `stock_knobs(rep)` names
    the lever in KNOB_LEVERS, the lever served no call and nothing fell back (its hook either answered every call with the stock statement
    — stock:path / stock:gate / stock:no_autocast / stock:dtype — or was never reached behind another idle lever). Such a lever gets
    e['knob'] = {'state': 'knob-off', 'flag': '<knob>:<value>'}: a by-design idle the user asked for through upstream's own flag —
    accounted (LEVER … state=skipped reason=stock_knob:<knob>:<value>), never partial (partial_of), exit 0. A lever with served 0 under
    DEFAULT knobs keeps the partial rule unchanged. Returns the evidence."""
    knobs = stock_knobs(rep)
    if not knobs:
        return evidence
    for knob, value in knobs.items():
        for lv in KNOB_LEVERS.get(knob, ()):
            e = evidence.get(lv)
            if not isinstance(e, dict) or e.get("served") or e.get("fallback") or e.get("knob"):
                continue
            e["knob"] = {"state": "knob-off", "flag": f"{knob}:{value}", "knob": knob, "value": value}
    return evidence


def partial_of(evidence: Dict[str, dict]) -> Tuple[List[str], Optional[str]]:
    """The PARTIAL state from `kit_evidence`: the levers of the mode with a fallback counter, or with no served call at all (no
    evidence of application; the gate counters say why). Returns (levers, one reason naming each lever's counters); ([], None) when
    every lever of the mode served with no fallback."""
    partial, why = [], []
    for lv, e in evidence.items():
        if e["fallback"]:
            partial.append(lv)
            why.append(f"{lv} fallback {e['fallback']}")
        elif not e["served"]:
            if (e.get("floor") or {}).get("state") == "floor-off":         # every item below the flash triangle-attention floor: the stock triangle attention by design
                continue
            if (e.get("rows") or {}).get("state") == "range-off":          # every call of the lever outside its served row range on this card: the stock forward by design
                continue
            if (e.get("gate") or {}).get("state") == "gate-off":           # every item at or below the lever's size gate: the stock statement by design (LEVER … state=skipped reason=below_gate)
                continue
            if (e.get("card") or {}).get("state") == "card-off":           # the lever has no build for this card / stack and stepped aside by name (LEVER … state=skipped reason=card_off): the stock statement by design
                continue
            if (e.get("aside") or {}).get("state") == "aside":              # a provider word whose row stepped aside by name for every call (another card, threshold, host absent): by design
                continue
            if (e.get("ceiling") or {}).get("state") == "ceiling-off":     # every item above the tri-attention ceiling: upstream row-chunks the attention, the stock statement by design (LEVER … state=skipped reason=above_ceiling)
                continue
            if (e.get("knob") or {}).get("state") == "knob-off":           # a stock kernel / dtype knob the user set routed the run off this lever's path (knob_states): the stock statement by the user's choice (LEVER … state=skipped reason=stock_knob:<knob>:<value>)
                continue
            partial.append(lv)
            why.append(f"{lv} served 0 calls (gated {e['gated'] or '{}'})")
    return partial, ("; ".join(why) or None)


def verdict(rep: dict, trimul: Optional[str], mode_levers, allow_partial: bool, run_ok: bool = True) -> dict:
    """The exit rule on a completed run: the activation-time partial state (`rep['partial']`: a lever the kit did not apply, admitted
    under the allowance) joined with the run-time state read from the kit's account (`rep['levers']`). Returns {partial, partial_reason,
    evidence, allow_partial, exit_code, items_failed}: partial and not allowed -> EXIT_NOT_ACTIVE; else EXIT_OK. `run_ok` False (the stock
    CLI's own non-zero exit) keeps the record and leaves the code to the caller (exit_code None). A failed or open item in the census
    (`rep['items_census']`: an item that raised inside `predict` — out of memory, … — or never returned) makes the run not ok by itself: the
    levers the failure cut short served no call by consequence, so the partial state is recorded but NOT judged (log_verdict prints the
    `NOTE exit rule not judged …` line in place of the partial line) and the exit code is the failure's (items_exit_code: 1)."""
    floor = (rep.get("triattn_floor") or {}).get("exported")
    census = rep.get("items_census") or {}
    items_failed = list(census.get("failed_items") or []) + ["?:open"] * int(census.get("open", 0) or 0)
    if items_failed:                                      # a failed or open item (an out-of-memory inside `predict`, …) IS the run's outcome: the levers past the
        run_ok = False                                    # failure served no call by consequence — the partial rule is not judged, the exit is the failure's (items_exit_code: 1)
    evidence = kit_evidence(rep.get("levers"), trimul, mode_levers, rep.get("items"), floor if isinstance(floor, int) else None) if (trimul and trimul != "stock") or mode_levers else {}
    evidence = knob_states(evidence, rep)                 # a stock kernel / dtype knob set by the user (--triatt_kernel / --trimul_kernel / --dtype): the levers it routes off the path idle BY NAME, exit 0
    partial, reason = partial_of(evidence)
    from . import big as B                              # big imports report: read at call time
    partial, reason = B.exit_join(partial, reason, evidence, bool(allow_partial), run_ok)   # the memory mode: base levers gated for the whole run at this size are disengaged by property (named, not partial); the memory levers' census gate joins the partial state; a no-op outside big
    act = list(rep.get("partial") or [])
    if act:
        partial = act + [p for p in partial if p not in act]
        reason = "; ".join(x for x in (rep.get("partial_reason"), reason) if x) or None
    code = None if not run_ok else (EXIT_NOT_ACTIVE if partial and not allow_partial else EXIT_OK)
    return {"partial": partial, "partial_reason": reason, "evidence": evidence, "allow_partial": bool(allow_partial), "exit_code": code,
            "items_failed": items_failed}


def log_verdict(v: dict, stream=None) -> Optional[str]:
    """The one line of the exit rule on a partial run (nothing on a full one): the exit form or the allowed form — or, on a run whose item
    census holds a failed / open item, the NOTE form (items_failed_note_line): the partial state is named but not judged, the exit is the failure's."""
    if not v.get("partial"):
        return None
    if v.get("items_failed"):
        return log(items_failed_note_line(v["partial"], v["items_failed"]), stream)
    line = partial_allowed_line(v["partial"], v.get("partial_reason")) if v.get("allow_partial") else partial_exit_line(v["partial"], v.get("partial_reason"))
    return log(line, stream)


def items_failed_note_line(partial: List[str], failed_items: List[str]) -> str:
    """`<PREFIX> NOTE exit rule not judged: <n> item(s) failed or open (<name:Type,…>) — <levers> served no call past the failure; the exit
    code is the failure's (1)` — the line in place of the partial line when an item raised inside `predict` (an out-of-memory names
    itself on the ITEM line) or never returned: a lever the failure cut short is a consequence, not a partial activation."""
    return note_line(f"exit rule not judged: {len(failed_items)} item(s) failed or open ({_token(','.join(failed_items))}) — "
                     f"{','.join(partial)} served no call past the failure; the exit code is the failure's ({EXIT_FAIL})")
