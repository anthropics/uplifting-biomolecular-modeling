"""Observability for protenix_opt: the activation line and the exit tally.

Two lines, both on stderr, both prefixed ``[protenix-opt]``:

* activation — ``ACTIVE mode=<m> protenix=<v> gpu=<name> levers=<a,b,...> fallbacks=<x,...>`` (or ``NOT ACTIVE: <reason>``),
* levers — one ``LEVER name=<strategy id> state=<on|off|skipped> [reason=<token>] impl=<module> origin=<core|kit> [k=v ...] mode=<m> lever=<registry name>``
  line per registry lever after ``FINAL`` (``lever_lines``; the shared grammar ``opt_core.report.lever_line``),
  formatted from the activation report returned by ``protenix_opt.enable()``;
* exit — the lever call counters of this process, read from the JSON line the kit appends to ``$PTX_LEVER_REPORT`` at interpreter
  exit (one line per process, keyed by ``pid``). The kit writes that line from ``ptx_trunk2_levers.report()`` in an ``atexit`` hook;
  ``register_exit_tally()`` must therefore be called BEFORE the levers are loaded so that (atexit being LIFO) the tally runs after the
  kit's hook. When no line for this pid exists the tally falls back to the in-memory counters of the loaded lever module and says so.
"""
from __future__ import annotations

import json
import os
import sys

from . import _core  # noqa: F401
from .registry import NOT_IN_ROW, STOCK_TRIATT
from opt_core import report as _core_report
from opt_core.mem import ngpu as _ngpu                          # the one producer of the `n_gpu=P sharding=<scheme>` token text
from opt_core.report import gpu_label, join as _join, log_once  # noqa: F401  (gpu_label: the tests read it here)

TAG = "protenix-opt"
PREFIX = "[protenix-opt]"                                  # the line prefix; == opt_core.report.prefix(TAG), test-locked

# Counter keys of the kit report line that are printed in the exit tally (names as written by ptx_trunk2_levers._STATS /
# ptx_fpf_v02.report(); absent keys are simply not printed). ``deadskip`` is a dict {n, n_expected, ...} and is printed as n/n_expected.
TALLY_KEYS = (
    "blk_calls", "blk_calls_chunked", "blk_fallback", "blk2_tri_calls", "blk_att_k2b_calls", "blk_att_k2b_routed",
    "trimul_partner_calls", "t1_fallback", "t2_calls", "zt_calls",
    "t5_calls", "og_calls", "og_fallback", "portability_lines",
   # the in-block transition's per-exact-M install check on t1_guard_by_m_arch arches (cc 8.0): first calls equal / unequal (-> M pinned to the module statement), later pinned calls, first calls met under capture
    "transition_core",                                            # transition_core_exact | transition_core: the provider binding's live record {word, calls, served, module, rows, module_reasons}
    "procuda",                                                    # triatt_prologue_cuda's live tally (PROCUDA_TALLY)
)


def partial_phrase(rep: dict) -> str:
    """`(partial on <GPU name>, cc <x.y>: kernel levers unavailable — a, b)` — the levers the kit's own applied/refused records left off."""
    g = rep.get("gpu") or {}
    return (f"(partial on {g.get('name')}, cc {g.get('cc')}: kernel levers unavailable — "
            f"{', '.join(rep.get('levers_unavailable') or rep.get('levers_fallback') or [])})")


def activation_line(rep: dict | None) -> str:
    """The one-line activation status for an activation report (the dict returned by protenix_opt.enable()/status()); a dry run
    (`check`) prints DRY-RUN with the levers the mode would apply, the number of variables it would export, the kernel key and the
    prebuilt fast-LN directory env.sh selected (`none` when no shipped build matches the installed torch). The ACTIVE line is the
    activation-time state: levers the kit installs later (stack.LATE_RECORDS) are reconciled at the end of the run (`final_line`)."""
    rep = rep or {}
    if rep.get("active"):
        if rep.get("partial"):                                    # the kit's own gates left levers off on this GPU: name them
            return f"{PREFIX} ACTIVE mode={rep.get('mode')} {ngpu_fields(rep)} {partial_phrase(rep)}{ablated_token(rep)}"
        return (f"{PREFIX} ACTIVE mode={rep.get('mode')} {ngpu_fields(rep)} protenix={rep.get('protenix_version')} gpu={gpu_label(rep.get('gpu'))} "
                f"levers={_join(rep.get('levers_applied'))} fallbacks={_join(rep.get('levers_fallback'))}{ablated_token(rep)}"
                + ("" if rep.get("reconciled") else " (levers installed later are reconciled at the end)"))
    reason = rep.get("reason") or "no reason given by protenix_opt.enable()"
    mode = rep.get("mode")
    if rep.get("dry_run") and rep.get("env") is not None:
        tg = rep.get("target_gpu")
        return (f"{PREFIX} DRY-RUN mode={mode} protenix={rep.get('protenix_version')} gpu={gpu_label(rep.get('gpu'))}"
                + (f" target_gpu={tg}" if tg else "")
                + f" levers={_join(rep.get('levers_applied'))} env_vars={len(rep.get('env') or {})}"
                + f" kernel_key={rep.get('kernel_key') or 'unknown'} prebuilt={rep.get('prebuilt') or 'none'}"
                + ablated_token(rep)
                + (f" {partial_phrase(rep)}" if rep.get("partial") else "")
                + (" notes=" + "; ".join(rep["notes"]) if rep.get("notes") else ""))
    return f"{PREFIX} NOT ACTIVE: {reason}" + (f" (mode={mode})" if mode else "")


def final_line(rep: dict) -> str:
    """The reconciled state at the end of a run (`pred`): the levers as the kit's own end-of-run records leave them."""
    rc = rep.get("reconciled") or {}
    moves = rc.get("moves") or {}
    return (f"{PREFIX} FINAL mode={rep.get('mode')} {ngpu_fields(rep)} levers={_join(rep.get('levers_applied'))} fallbacks={_join(rep.get('levers_fallback'))}"
            f"{ablated_token(rep)} partial={str(bool(rep.get('partial'))).lower()} reconciled_with={_join(rc.get('records'))}"
            + (" moves=" + ",".join(f"{k}:{v.replace(' ', '')}" for k, v in moves.items()) if moves else "")
            + templates_phrase(rep))


def ablated_token(rep: dict | None) -> str:
    """`` ablated=<lever,…>`` when ``MODEL_OPT_LEVERS_OFF`` removed levers from the mode for this run (``rep["levers_ablated"]``, the
    ablation module's token); the empty string otherwise, so an un-ablated run's lines read exactly as before."""
    names = list((rep or {}).get("levers_ablated") or [])
    return f" ablated={','.join(names)}" if names else ""


def ngpu_fields(rep: dict | None = None) -> str:
    """``n_gpu=P sharding=<scheme>`` — the GPU-count tokens every ACTIVE / FINAL / EXIT line carries (the shared core's text,
    ``opt_core.mem.ngpu.active_fields``): ``n_gpu=1 sharding=none`` for a single-GPU process (every mode's default), ``n_gpu=P
    sharding=rowpair`` in the parent of the multi-GPU line (``tp``; the report carries ``n_gpu`` / ``sharding``, else ``RUN``)."""
    rep = rep or {}
    n = int(rep.get("n_gpu") or RUN.get("n_gpu") or 1)
    return _ngpu.active_fields(n, rep.get("sharding") or RUN.get("sharding") or "rowpair")


RUN: dict = {"n_gpu": 1, "sharding": "rowpair"}                       # the process's GPU count for the EXIT line (cli's multi-GPU route sets n_gpu = P; every other process is 1)


def templates_phrase(rep: dict) -> str:
    """The template census guard's evidence on the FINAL line (`protenix_opt/templates.py`): ` templ_items=<n> templ_chains=<n>
    templ_hits=<n> templ_real=<n> templ_dropped=<n> templ_all_dummy=<n>` — present whenever the guard featurised an item; empty otherwise."""
    t = rep.get("templates") or {}
    if not t or not t.get("items"):
        return ""
    return (f" templ_items={t.get('items', 0)} templ_chains={t.get('chains_requested', 0)} templ_hits={t.get('hits', 0)}"
            f" templ_real={t.get('real', 0)} templ_dropped={t.get('dropped', 0)} templ_all_dummy={t.get('all_dummy_items', 0)}")


# The canonical strategy id of every registry lever (`common/opt_core/opt_core/STRATEGIES.json`: F1 triangle attention, F2 triangle multiplication,
# F3 capture, F4 precision, F5 attention / gates, F6 host, F7 memory; LOCAL.<name> shared-local, LOCAL.protenix_v2.<name> this kit's own) — the `name=`
# and `strategy=` of the per-lever LEVER line (opt_core.report.strategy_form checks the form; tests/test_strategy_catalogue.py the catalogue
# membership); the registry name rides as `lever=`.
STRATEGY_IDS = {
    "layernorm_fast": "LOCAL.layernorm_kernel",
    "cueq_tuned_tiles": "F2.cueq_tuned_tiles",
    "template_dedupe": "LOCAL.protenix_v2.template_dedupe",
    "t1_fused_transition": "LOCAL.fused_transition",
    "nomask": "F1.triatt_lean_exact",
    "blk2_block_path": "F1.triatt_lean_exact",
    "pwa_zcache": "LOCAL.protenix_v2.msa_pwa_zcache",
    "blk2_chunked_exact": "F7.chunked_eval",
    "deadskip": "LOCAL.deadskip",
    "lever_report": "LOCAL.protenix_v2.lever_report",
    "trimul_core_exact": "F2.fpf_trimul_exact", "trimul_core": "F2.fpf_trimul_fast",   # the pair-stack TriMul through opt_core.kernels.trimul by tier word (src/ptx_trimul_routes.py)
    "stackgraph": "F3.cuda_graph_trunk",
    "sampler_graph": "F3.cuda_graph_sampler",
    "sampler_graph_cache_policy": "F3.graph_admission_policy",
    "sampler_prep": "LOCAL.protenix_v2.sampler_prep", "sampler_reach": "F3.graph_admission_policy",
    "sampler_fuse": "LOCAL.dit_fused_kernels",
    "fastln_prebuilt": "LOCAL.layernorm_kernel",
    "ln_core": "LOCAL.layernorm_kernel",
    "xl_policy": "F7.chunked_eval",
    "pad8": "F5.kernel_glue_gates",
    "glue_v2": "F5.kernel_glue_gates",
    "mk_pf": "LOCAL.protenix_v2.mk_pairformer",
    "guard_lift": "LOCAL.protenix_v2.guard_lift",
    "lazy_init": "F6.weights_residency_init",
    "blk2_chunked_k2b": "F1.flash_triatt",
    "k2b_flash_triattention": "F1.flash_triatt",
    "smalln_size_gate": "F5.kernel_glue_gates",
    "drop_bond_mask": "LOCAL.protenix_v2.drop_bond_mask",
    "cond_chunk": "F7.chunked_eval",
    "apb_bias_chunk": "F7.chunked_eval",
    "relp_lazy": "F7.chunked_eval",
    "msa_zfree": "F7.chunked_eval",
    "diffcache_free": "F7.chunked_eval",
    "cache_release": "LOCAL.protenix_v2.cache_release",
    "sampler_admit": "LOCAL.protenix_v2.sampler_admit",
    "pred_release": "LOCAL.protenix_v2.pred_release",
    "keep_pool": "LOCAL.protenix_v2.keep_pool",
    "summary_hostidx": "F6.host_sync_elimination",
    "dit_attn_exact": "LOCAL.protenix_v2.dit_attn_exact",
    "triatt_exact": "LOCAL.protenix_v2.triatt_exact",
    "templ_trimul_tmk3": "F2.fpf_trimul_exact", "templ_trimul_esm": "F2.fpf_trimul_fast", "templ_triatt_core": "F1.flash_triatt",                     # the template embedder's c = 64 pair stack through the shared core's providers (src/ptx_c64_routes.py)
    "transition_core_exact": "LOCAL.fused_transition", "transition_core": "LOCAL.fused_transition",                                  # opt_core.kernels.transition by tier word (src/ptx_transition_core.py)
    "triatt_prologue_cuda": "LOCAL.protenix_v2.triatt_prologue_cuda",
    "pf_attn": "LOCAL.protenix_v2.pf_attn", "opm_fused": "LOCAL.protenix_v2.opm_fused", "pwa_fused": "LOCAL.protenix_v2.pwa_fused",
    "cond_dedupe": "LOCAL.protenix_v2.cond_dedupe", "dit_fused": "LOCAL.protenix_v2.dit_fused", "dit_lowp": "F4.autocast_policy",   # sampler_levers.STRATEGIES
    "atom_fused": "LOCAL.protenix_v2.atom_fused", "atom_attn_exact": "LOCAL.protenix_v2.atom_attn_exact",
    "triattn_native": "F1.flash_triatt",
    "dit_attn": "F5.flash_attn_dense",
    "dit_attn_fp16": "F4.autocast_policy",
    "atom_attn": "F5.flash_attn_dense",
}
# (impl, origin) of every registry lever: the module, package or switch-table site that implements it; origin core = served from common/opt_core.
IMPL = {
    "layernorm_fast": ("stock:protenix/model/layer_norm", "kit"), "cueq_tuned_tiles": ("forward/flashpairformer/levers_addon/PTXV2_LEVERS_ADDON_v1/cueq_cache", "kit"),
    "template_dedupe": ("forward/flashpairformer/levers_addon/PTXV2_LEVERS_ADDON_v1/ptx_addon_levers.py", "kit"),
    "t1_fused_transition": ("opt_core/kernels/fpf_transition", "core"), "nomask": ("forward/flashpairformer/src/ptx_trunk2_levers.py", "kit"),
    "blk2_block_path": ("opt_core/kernels/fpf_triatt_pro", "core"), "pwa_zcache": ("forward/flashpairformer/src/ptx_trunk2_levers.py", "kit"),
    "blk2_chunked_exact": ("forward/flashpairformer/src/ptx_trunk2_levers.py", "kit"), "deadskip": ("forward/flashpairformer/src/deadskip.py", "kit"),
    "lever_report": ("forward/flashpairformer/src/ptx_fpf_v02.py", "kit"), "trimul_core_exact": ("opt_core/kernels/trimul", "core"), "trimul_core": ("opt_core/kernels/trimul", "core"),
    "stackgraph": ("forward/flashpairformer/src/fpf_stackgraph", "kit"), "sampler_graph": ("forward/flashpairformer/src/fpf_clisampler", "kit"),
    "sampler_graph_cache_policy": ("forward/flashpairformer/src/infopt_graphs", "kit"),
    "sampler_prep": ("forward/flashpairformer/src/infopt_graphs", "kit"), "sampler_reach": ("forward/flashpairformer/src/infopt_graphs", "kit"),
    "sampler_fuse": ("forward/DIT_FUSE/tools/dit_fuse_patch.py", "kit"),
    "fastln_prebuilt": ("forward/flashpairformer/third_party/fastln_prebuilt", "kit"), "xl_policy": ("forward/flashpairformer/src/ptx_trunk2_levers.py", "kit"),
    "ln_core": ("opt_core.kernels.ln", "core"),                                            # bound by forward/flashpairformer/src/protenix_ptx_ln_core.py (the kit's provider binding); the rows are the core's
    "pad8": ("forward/flashpairformer/third_party/fpf_pad8exact", "kit"), "glue_v2": ("opt_core/kernels/fpf_glue_v2", "core"),
    "mk_pf": ("opt_core/kernels/fpf_mkpf", "core"),
    "guard_lift": ("protenix_opt/runner_hooks.py", "kit"),
    "lazy_init": ("forward/flashpairformer/src/ptx_lazy_init", "kit"),
    "blk2_chunked_k2b": ("opt_core/kernels/fpf_triatt_k2b", "core"), "k2b_flash_triattention": ("opt_core/kernels/fpf_triatt_k2b", "core"),
    "smalln_size_gate": ("forward/flashpairformer/third_party/fpf_smalln", "kit"),
    "drop_bond_mask": ("protenix_opt/big.py", "kit"), "cond_chunk": ("protenix_opt/big.py", "kit"), "apb_bias_chunk": ("protenix_opt/big.py", "kit"),
    "cache_release": ("opt_core.mem.allocator", "core"),
    "sampler_admit": ("forward/flashpairformer/src/fpf_clisampler/policy.py", "kit"), "pred_release": ("protenix_opt/pred_release.py", "kit"),
    "relp_lazy": ("protenix_opt/big.py", "kit"), "msa_zfree": ("protenix_opt/big.py", "kit"), "diffcache_free": ("protenix_opt/big.py", "kit"),
    "keep_pool": ("forward/flashpairformer/src/protenix_ptx_keep_pool.py", "kit"), "summary_hostidx": ("forward/flashpairformer/src/protenix_ptx_summary_host.py", "kit"),
    "triattn_native": ("opt_core.kernels.triattn", "core"),                    # bound by forward/flashpairformer/src/ptx_native_core.py (the kit's provider slot); the kernel is the core's sealed package
    "dit_attn_exact": ("opt_core.kernels.apb", "core"),   # dit_attn_exact: protenix_opt/apb_core.py, tier word exact -> the provider's dit_exact row where vouched on the stack
    "triatt_exact": ("opt_core.kernels.triattn", "core"),   # triatt_exact: forward/flashpairformer/src/fpf/triatt_exact.py, tier word exact -> the provider's triattn_exact row where vouched on the stack, the library op by name elsewhere
    "templ_trimul_tmk3": ("opt_core/kernels/trimul", "core"), "templ_trimul_esm": ("opt_core/kernels/trimul_esm_shapes", "core"), "templ_triatt_core": ("opt_core/kernels/triattn", "core"),
    "transition_core_exact": ("opt_core/kernels/transition", "core"), "transition_core": ("opt_core/kernels/transition", "core"),
    "triatt_prologue_cuda": ("forward/flashpairformer/third_party/protenix_fpf_triatt_procuda", "kit"),
    "pf_attn": ("opt_core.kernels.apb", "core"),                                      # bound by protenix_opt/apb_core.py by the mode's tier word (producer pair_bias_planes + core pair_bias_attention)
    "opm_fused": ("forward/flashpairformer/third_party/protenix_fpf_msa", "kit"), "pwa_fused": ("forward/flashpairformer/third_party/protenix_fpf_msa", "kit"),
    "cond_dedupe": ("forward/flashpairformer/third_party/protenix_fpf_ditfast", "kit"), "dit_fused": ("forward/flashpairformer/third_party/protenix_fpf_ditfast", "kit"),
    "dit_lowp": ("forward/flashpairformer/third_party/protenix_fpf_ditfast", "kit"), "atom_fused": ("forward/flashpairformer/third_party/protenix_fpf_ditfast", "kit"),
    "atom_attn_exact": ("opt_core.kernels.apb", "core"),                   # served from the shared core (opt_core.kernels.apb)
    "dit_attn": ("opt_core.kernels.apb", "core"), "dit_attn_fp16": ("opt_core.kernels.apb", "core"),   # bound by protenix_opt/apb_core.py by the mode's tier word, every card
    "atom_attn": ("opt_core.kernels.apb", "core"),
}
OFF_BY_FLAG = "off by flag"                        # the head of stack._classify's reason for a lever the caller removed by its switch (PTX_SAMPLER_FUSE=0)
ABLATED = "ablated"                                   # the LEVER line's reason for a lever MODEL_OPT_LEVERS_OFF removed from the mode for this run (ablation.REASON)


CUEQ_TILE_TABLE = "fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper"      # cuequivariance_ops' TriMul (fused sigmoid-gated dual GEMM) Triton autotune table: <name>.<cc>.json under $CUEQ_TRITON_CACHE_DIR


def cueq_tiles_dir(env=None) -> str:
    """The directory cuequivariance_ops reads its TriMul tile table from in a kit-mode process: ``$CUEQ_TRITON_CACHE_DIR`` when set (env.sh L7
    points it at the levers add-on's ``cueq_cache/``), else that kit directory itself."""
    env = os.environ if env is None else env
    d = (env.get("CUEQ_TRITON_CACHE_DIR") or "").strip()
    if d:
        return d
    from .stack import kit_home                                   # lazy: stack imports this module at load
    return os.path.join(kit_home(), "levers_addon", "PTXV2_LEVERS_ADDON_v1", "cueq_cache")


def cueq_tiles_evidence(rep: dict | None = None, env=None) -> list:
    """The ``cueq_tuned_tiles`` LEVER pairs — a statement about files, so the line never claims a tile table that is not there:
    ``tiles=table_sm<NN>`` only when ``<cueq_tiles_dir>/<CUEQ_TILE_TABLE>.<cc>.json`` exists for the running card (``<cc>`` = its compute
    capability; a cc 10.3 / 10.7 card reads the cc 10.0 table — cuequivariance_ops' own rule — and the pair names the file's sm), in which case
    the library serves tile choices from that file as a pure lookup and never opens its packaged table; else
    ``tiles=packaged_defaults(no_table_sm<NN>)`` — the library's own packaged per-card table applies, the same tiles the stock route reads.
    ``tiles=unknown(no_cc)`` when the report names no GPU."""
    gpu = (rep or {}).get("gpu") or {}
    cc = str(gpu.get("cc") or "").strip()
    if not cc:
        return [("tiles", "unknown(no_cc)")]
    major, _, minor = cc.partition(".")
    file_cc = f"{major}.0" if (major == "10" and minor in ("3", "7")) else cc
    sm = "sm" + file_cc.replace(".", "")
    path = os.path.join(cueq_tiles_dir(env), f"{CUEQ_TILE_TABLE}.{file_cc}.json")
    return [("tiles", f"table_{sm}" if os.path.isfile(path) else f"packaged_defaults(no_table_{sm})")]


def _token(v) -> str:
    """A value with no blank in it (readers split the LEVER line on blanks)."""
    return str(v).strip().replace(" ", "_")


def _evidence(name: str, rep: dict) -> list:
    """The evidence pairs of a lever that is on, from the package's own records in the report (empty for the kit's vendor levers)."""
    if name == "cueq_tuned_tiles":
        return cueq_tiles_evidence(rep)                          # which tile table the library reads on this card — from the files present, never a claim
    if name == "sampler_fuse":
        from . import sampler_fuse
        return sampler_fuse.evidence()
    if name in ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact"):
        from . import sampler_levers
        return sampler_levers.evidence(name)
    if name in ("dit_attn", "dit_attn_fp16", "atom_attn", "pf_attn", "opm_fused", "pwa_fused"):
        from . import apb_levers
        return apb_levers.evidence(name)
    if name in TAIL_UNITS:
        return tail_evidence(name)
    if name == "triattn_native":
        return native_evidence()
    if name == "dit_attn_exact":
        return dit_attn_exact_evidence()
    if name == "triatt_exact":
        return triatt_exact_evidence(rep)
    if name in ("trimul_core", "trimul_core_exact"):
        return trimul_routes_evidence()
    if name in ("transition_core_exact", "transition_core"):
        return transition_core_evidence(rep)
    if name == "triatt_prologue_cuda":
        return procuda_evidence()
    if name == "ln_core":
        return ln_core_evidence()
    if name == "sampler_reach":
        return sampler_reach_evidence(rep)
    if name == "sampler_admit":
        return sampler_admit_evidence(rep)
    if name == "pred_release":
        from . import pred_release as _pr
        return _pr.evidence()
    if name == "sampler_prep":
        return sampler_prep_evidence(rep)
    rh = rep.get("runner_hooks") or {}
    if name == "guard_lift" and rh:
        return [("lift", int(bool(rh.get("lift")))), ("items", len(rh.get("items") or []))]
    from .modes import MEM_LEVERS
    if name in MEM_LEVERS and rep.get("big"):
        from . import big
        return big.evidence(name, rep["big"])
    return []


TAIL_UNITS = {"keep_pool": ("protenix_ptx_keep_pool", ("skipped_total", "passed_total", "errors")),                 # lever -> (module, the report() keys its LEVER line carries)
              "summary_hostidx": ("protenix_ptx_summary_host", ("calls", "samples", "delegated", "chains_max", "pairs", "d2h", "h2d", "errors"))}


def tail_evidence(name: str) -> list:
    """keep_pool / summary_hostidx LEVER pairs, read live from the unit's own report() in this process (the same counters its SUMMARY line and
    $PTX_LEVER_REPORT record carry): keep_pool — skipped_total (stock forward-path empty_cache calls made no-ops: one per item at confidence.py:204,
    + N_sample per item above 2000 tokens), passed_total (kit / runner releases passed through), errors; summary_hostidx — calls (items), samples,
    delegated (non-inference call shapes handed to the stock function: 0 on pred), chains_max, pairs, d2h / h2d copies, errors. ``unit=none`` when
    the module is not loaded in this process (a dry run, a launcher)."""
    modname, keys = TAIL_UNITS[name]
    mod = sys.modules.get(modname)
    if mod is None or not hasattr(mod, "report"):
        return [("unit", "none")]
    try:
        r = dict(mod.report() or {})
    except Exception as e:  # noqa: BLE001 — a report that raises is written on the line, not fatal at exit
        return [("unit", "report_error"), ("error", _token(repr(e))[:80])]
    if not r.get("installed"):                                   # imported but never installed here (a CPU test, a launcher): nothing ran through it
        return [("unit", "none")]
    return [("installed", 1)] + [(k, int(r.get(k) or 0)) for k in keys]


PROCUDA_TALLY = ("cuda", "rm", "hm", "pad", "pass_shape", "pass_wx", "chk_eq", "chk_ne", "chk_defer", "chk_pending")


def ln_core_evidence() -> list:
    """ln_core's LEVER pairs, read live from the binding unit in this process (src/protenix_ptx_ln_core.report()): installed, word (the tier word asked), core (the opt_core
    version bound), calls (FusedLayerNorm.forward calls), served (executed by a provider row), rows (arm:calls,... or none), byname (left on the module's
    fast_layernorm by name) and its split byname_form / byname_stock / byname_uncovered / byname_refused, classes (distinct decisions), uncovered (cell classes with no
    family for the width, or none)."""
    import sys as _sys
    m = _sys.modules.get("protenix_ptx_ln_core")
    r = m.report() if m is not None else {}
    if not r.get("installed"):                                     # not bound in this process (unit absent or not installed): one pair, whatever imported the unit
        return [("installed", 0)]
    rows = ",".join(f"{_tok(k)}:{v}" for k, v in sorted((r.get("rows") or {}).items())) or "none"
    unc = ",".join(sorted(r.get("uncovered") or {})) or "none"
    return [("installed", int(bool(r.get("installed")))), ("word", r.get("word") or "none"), ("core", _token(r.get("opt_core") or "none")), ("calls", r.get("calls", 0)),
            ("served", r.get("core", 0)), ("rows", rows), ("byname", r.get("byname", 0)), ("byname_form", r.get("byname_form", 0)), ("byname_stock", r.get("byname_stock", 0)),
            ("byname_uncovered", r.get("byname_uncovered", 0)), ("byname_refused", r.get("byname_refused", 0)), ("classes", r.get("classes", 0)), ("uncovered", _token(unc))]


def procuda_evidence() -> list:
    """triatt_prologue_cuda's LEVER pairs, read live from the trunk levers' tally (``ptx_trunk2_levers._STATS["procuda"]``, the dict the unit
    publishes at install and the EXIT line's ``procuda=`` token prints): cuda = node calls served by the kernel (= rm + hm), pad = of which into
    padded buffers, pass_shape / pass_wx = calls outside the envelope sent to the F1 cell by declared reason, chk_eq / chk_ne = first-call shape
    checks (an chk_ne never survives: it raises by name), chk_defer = shapes first met under graph capture, chk_pending = deferred shapes not yet
    checked (0 at exit). ``tally=none`` when the trunk levers did not run in this process."""
    mod = sys.modules.get("ptx_trunk2_levers")
    t = (getattr(mod, "_STATS", None) or {}).get("procuda") if mod is not None else None
    if not isinstance(t, dict):
        return [("tally", "none")]
    return [(k, int(t.get(k) or 0)) for k in PROCUDA_TALLY]


def transition_core_evidence(rep: dict | None) -> list:
    """transition_core_exact | transition_core LEVER pairs off the binding's live record (ptx_trunk2_levers report()['transition_core'] = ptx_transition_core.report()):
    word (the tier word bound), bind=tier:<word>, calls, served (calls the provider's rows served) with rows=<row@cell:count,...>, module (calls the forward found at
    install answered by name) with module_reasons=<reason:count,...>, opt_core=<version>; ``unit=none`` when the binding never installed in this process."""
    r = None
    fpf = (rep or {}).get("fpf") if isinstance(rep, dict) else None
    if isinstance(fpf, dict):
        r = fpf.get("transition_core")
    if r is None:
        try:
            mod = sys.modules.get("ptx_transition_core")
            r = mod.report() if mod is not None else None
        except Exception as e:  # noqa: BLE001
            return [("unit", "report_error"), ("error", _token(repr(e))[:80])]
    if not isinstance(r, dict) or not r.get("installed"):
        return [("unit", "none")]
    rows = r.get("rows") if isinstance(r.get("rows"), dict) else {}
    why = r.get("module_reasons") if isinstance(r.get("module_reasons"), dict) else {}
    return [("word", str(r.get("word"))), ("bind", "tier:%s" % r.get("word")), ("calls", int(r.get("calls") or 0)), ("served", int(r.get("served") or 0)),
            ("rows", _token(",".join(f"{k}:{v}" for k, v in sorted(rows.items()))) if rows else "none"), ("module", int(r.get("module") or 0)),
            ("module_reasons", _token(",".join(f"{k}:{v}" for k, v in sorted(why.items()))) if why else "none"), ("opt_core", _token(str(r.get("opt_core") or "?")))]


SAMPLER_PREP_STATS = ("key_shape_miss", "key_value_miss", "pool_chained", "pool_renewed", "poison_skipped", "rot_ring_waits")


def sampler_reach_evidence(rep: dict | None) -> list:
    """sampler_reach's LEVER pairs off the clisampler record: reach (the record's word: on(floor=...) | off), and the reach_* route counters of prep_stats
    (reach_graph / reach_hoist_eager / reach_stock / reach_released: items above the floor by the route the policy admitted)."""
    g = ((rep or {}).get("gates") or {}).get("clisampler") or {}
    if not g:
        return [("record", "none")]
    st = g.get("prep_stats") if isinstance(g.get("prep_stats"), dict) else {}
    word = str(g.get("reach") or "absent")
    pairs = [("reach", "on" if word.startswith("on(") else _token(word)[:40]), ("floor", _token(word.split("floor=")[1].split(";")[0]) if "floor=" in word else "-")]
    pairs += [(k, int(st.get(k) or 0)) for k in ("reach_graph", "reach_hoist_eager", "reach_stock", "reach_released")]
    return pairs


def sampler_admit_evidence(rep: dict | None) -> list:
    """sampler_admit's LEVER pairs off the clisampler record's policy summary: words (admit/release/hoist_eager), the admission model's version, verdicts
    per route, releases / released_gib, misses (an ADMIT_MISS is printed on its SAMPLER-MEM line and counted here; 0 on a fitted card)."""
    g = ((rep or {}).get("gates") or {}).get("clisampler") or {}
    pol = g.get("policy") if isinstance(g.get("policy"), dict) else None
    if pol is None:
        return [("record", "none")]
    v = pol.get("verdicts") if isinstance(pol.get("verdicts"), dict) else {}
    model = pol.get("model") if isinstance(pol.get("model"), dict) else {}
    mod = sys.modules.get("fpf_clisampler.policy")
    words = str(pol.get("words") or "-").replace("=", ":").replace(" ", ",")                 # admit:item,release:item,hoist_eager:1 (one whitespace-free token, no '=' inside the value)
    return [("words", _token(words)[:60]), ("model", _token(str(pol.get("version") or getattr(mod, "POLICY_VERSION", None) or model.get("version") or "-"))),
            ("graph", int(v.get("graph") or 0)), ("hoist_eager", int(v.get("hoist_eager") or 0)), ("stock", int(v.get("stock") or 0)),
            ("releases", int(pol.get("releases") or 0)), ("released_gib", _token(str(pol.get("released_gib") or 0))), ("misses", int(pol.get("admit_misses") or 0))]


def sampler_prep_evidence(rep: dict | None) -> list:
    """sampler_prep's LEVER pairs off the clisampler end-of-run record the reconciled report carries under ``gates.clisampler``: parts = the
    prep= word (all six parts, the only accepted word), poison = the once-per-process self-test's verdict, then the counters that say the parts
    ran (shape misses, value misses, pools chained / renewed, poison probes skipped, rotation-ring waits); ``record=none`` before the record exists."""
    g = ((rep or {}).get("gates") or {}).get("clisampler") or {}
    if "prep" not in g:
        return [("record", "none")]
    out = [("parts", str(g.get("prep") or "off").split(";")[0].strip()), ("poison", g.get("prep_poison") or "not-run")]
    st = g.get("prep_stats") if isinstance(g.get("prep_stats"), dict) else {}
    return out + [(k, int(st.get(k) or 0)) for k in SAMPLER_PREP_STATS]


def trimul_routes_evidence() -> list:
    """trimul_core / trimul_core_exact LEVER pairs, read live from src/ptx_trimul_routes.py (its Lever's census): word / bind / served / fallback
    (the per-reason stock-forward counts folded into one ``reason:n,…`` token) / passthrough (calls the module hands to the stock forward: c != 256,
    the torch branch). ``tally=none`` when the module did not run in this process (a dry run, a launcher)."""
    mod = sys.modules.get("ptx_trimul_routes")
    if mod is None:
        return [("tally", "none")]
    try:
        rep = mod.report()
    except Exception as e:  # noqa: BLE001
        return [("tally", "unreadable:" + _tok(repr(e))[:60])]
    c = rep.get("census") or {}
    fb = c.get("fallback") or {}
    fbt = ",".join(f"{_tok(str(k))}:{int(v)}" for k, v in sorted(fb.items())) if isinstance(fb, dict) and fb else ("0" if isinstance(fb, dict) else _tok(str(fb)))
    return [("word", _tok(rep.get("word"))), ("bind", _tok(rep.get("bind"))), ("served", int(c.get("served") or 0)), ("fallback", fbt), ("passthrough", int(rep.get("passthrough") or 0))]


def _tok(x: str) -> str:
    """One whitespace-free token."""
    return "".join(ch if ch not in " \t\n" else "_" for ch in x)


def lever_lines(rep: dict | None) -> list[str]:
    """One ``LEVER`` line per lever the kit knows (``registry.LEVERS`` order) for this process, in the shared per-lever grammar
    (``opt_core.report.lever_line``): ``[protenix-opt] LEVER name=<strategy id> state=<on|off|skipped> [reason=<token>] impl=<module>
    origin=<core|kit> strategy=<canonical id> [evidence k=v ...] mode=<m> lever=<registry name>``. From a reconciled activation report: ``on`` = applied and live in
    this process (``levers_applied``); ``skipped`` with a one-token reason — ``fallback`` (+ ``fallback_by=<the kit's reason>``:
    ``levers_fallback``), ``off_by_flag`` (a lever the caller's switch removed: ``PTX_SAMPLER_FUSE=0``), ``not_in_arm`` (+ ``why``: a lever outside this arm on this
    kernel key), ``unaccounted`` (a mode lever the report leaves in no list); ``off`` with ``reason=ablated`` = removed from the mode
    for this run by ``MODEL_OPT_LEVERS_OFF``;
    ``off`` with ``reason=not_in_mode:<m>`` = not in this mode's lever set. An inactive report yields no lines."""
    rep = rep or {}
    if not rep.get("active") or not rep.get("mode"):
        return []
    from .registry import LEVERS
    from .modes import MODES, big_levers
    mode = rep.get("mode")
    in_mode = set(big_levers() if mode == "big" else (MODES.get(mode) or []))   # big's lever set is its base's composition
    on, fb = set(rep.get("levers_applied") or []), set(rep.get("levers_fallback") or [])
    nia, why = set(rep.get("levers_not_in_arm") or []), dict(rep.get("fallback_reasons") or {})
    abl = set(rep.get("levers_ablated") or [])                  # MODEL_OPT_LEVERS_OFF: removed from the mode for this run, written as off / ablated
    subsumed = census_rebase(on)                                 # {subsumed lever: [(by, counter, expected), …]} from the levers ON in this process
    lines = []
    for name, lv in LEVERS.items():
        impl, origin = IMPL[name]
        reason, pairs = None, []
        if name in on:
            state, pairs = "on", _evidence(name, rep)
            pairs = pairs + subsumption_pairs(name, pairs, subsumed) + ([("subsumes", ",".join(f"{l}:{c}" for l, c, _ in lv.subsumes))] if lv.subsumes else [])
        elif name in fb:
            state, reason, pairs = "skipped", "fallback", [("fallback_by", _token(why.get(name) or "no_reason_recorded"))]
        elif name in nia:
            if str(why.get(name) or "").startswith(OFF_BY_FLAG):     # a memory lever removed from the line by its flag: a recorded opt-out
                state, reason = "skipped", "off_by_flag"
            else:
                state, reason, pairs = "skipped", "not_in_arm", [("why", _token(why.get(name) or "no_reason_recorded"))]
        elif name in abl:
            state, reason = "off", ABLATED
        elif name in in_mode:                                    # a mode lever the report leaves in no list: never written as off
            state, reason = "skipped", "unaccounted"
        else:
            state, reason = "off", f"not_in_mode:{mode}"
        sid = STRATEGY_IDS[name]
        lines.append(_core_report.lever_line(TAG, sid, state, *pairs, reason=reason, impl=impl, origin=origin, strategy=sid, mode=mode, lever=name))
    return lines




def native_state() -> dict:
    """ptx_native_core's report() when the provider module is loaded in this process ({} otherwise): install record (opt_core version, cc, key, word,
    bind, bind_note, native_pkg, select = the rows the tier word names at the probe sizes, describe), counts {calls, native, other_row, refused}, rows,
    fallback_rows, refused_kinds, forms, sizes, pkg_fallbacks."""
    mod = sys.modules.get("ptx_native_core")
    if mod is None or not hasattr(mod, "report"):
        return {}
    try:
        return dict(mod.report())
    except Exception as e:  # noqa: BLE001
        return {"report_error": repr(e)}


def native_evidence() -> list:
    """The LEVER line pairs of triattn_native: installed, the MODE's tier word (word=fast|big), prefer=none (no row preference), bind = the tier word handed
    to the shared core's face literally (tier:fast | tier:big; opt_core >= 0.5.114.0), the core's
    version and ABI key, the sealed package version the core carries (native_pkg), min_tokens=0 (no kit floor), rows = provider-served calls BY THE ROW THAT
    SERVED THEM (triattn_native | cuda_sm90a | k2b | ...: which kernel actually ran), fallback_rows = calls a row refused by name, by the row the provider named
    instead, the refusal kinds, call forms, size buckets and the package's counted operand copies."""
    r = native_state(); c = dict(r.get("counts") or {})
    kinds = ",".join(f"{k}:{n}" for k, n in sorted((r.get("refused_kinds") or {}).items())) or "none"
    sizes = ",".join(f"{k}:{n}" for k, n in (r.get("sizes") or {}).items()) or "none"
    rows = ",".join(f"{k}:{n}" for k, n in sorted((r.get("rows") or {}).items())) or "none"
    fbrows = ",".join(f"{k}:{n}" for k, n in sorted((r.get("fallback_rows") or {}).items())) or "none"
    forms = ",".join(f"{k}:{n}" for k, n in sorted((r.get("forms") or {}).items())) or "none"
    copies = ",".join(f"{k}:{n}" for k, n in sorted((r.get("pkg_fallbacks") or {}).items()) if n) or "none"
    return [("installed", int(bool(r.get("installed")))), ("word", r.get("word") or "none"), ("prefer", "none"), ("bind", str(r.get("bind") or "none")),
            ("core", _token(str(r.get("opt_core") or "none"))), ("key", str(r.get("key") or "none").replace(" ", "")),
            ("native_pkg", str(r.get("native_pkg") or "none").replace(" ", "_")), ("min_tokens", 0), ("rows", rows), ("fallback_rows", fbrows), ("forms", forms),
            ("calls", int(c.get("calls") or 0)), ("native", int(c.get("native") or 0)), ("other_row", int(c.get("other_row") or 0)), ("refused", kinds), ("sizes", sizes), ("pkg_copies", copies)]


def dit_attn_exact_evidence() -> list:
    """dit_attn_exact's LEVER pairs, read live from the unit's report() in this process: installed, calls (Python-level route decisions: under the
    sampler graph the warm-up + capture calls; the replayed graphs contain the kernel), kernel = calls served by the prebuilt kernel, other = calls
    that took the original statement by declared route (atom transformer head dim, bf16 sites …), loadcheck cases. ``unit=none`` when the module is
    not loaded here (a dry run, a launcher, a row that does not list the lever)."""
    mod = sys.modules.get("protenix_opt.apb_core")                 # the tier-word binding (protenix_opt.apb_core) holds the hook's record
    if mod is None or not hasattr(mod, "report"):
        return [("unit", "none")]
    try:
        r = dict(mod.report() or {})
    except Exception as e:  # noqa: BLE001
        return [("unit", "report_error"), ("error", _token(repr(e))[:80])]
    if not r.get("installed"):
        return [("unit", "none")]
    routes = dict(r.get("routes") or {}); k = int(routes.pop("kernel", 0) or 0)
    lc = r.get("loadcheck") or {}
    pairs = [("installed", 1), ("calls", int(r.get("calls") or 0)), ("kernel", k), ("other", int(sum(int(v or 0) for v in routes.values()))),
             ("loadcheck", (f"{lc.get('cases', 0)}cases" + ("-bit-equal" if lc.get("ok", lc.get("bit_equal", True)) else "-MISMATCH")) if lc else "none")]
    if r.get("row"):                                                 # the provider row the exact word resolved to on this stack (dit_exact | statement) -- trailing pairs
        pairs += [("row", _token(r["row"])), ("word", "exact")]
    return pairs


def triatt_exact_state(rep: dict | None = None) -> dict:
    """triatt_exact's record: the trunk record's entry (ptx_trunk2_levers report()['triatt_exact'] = fpf.triatt_exact.report()) when the report
    carries it, else the binding module's live report() in this process; {} when neither exists (a dry run, a launcher, a row without the lever)."""
    fpf = (rep or {}).get("fpf") if isinstance(rep, dict) else None
    if isinstance(fpf, dict) and isinstance(fpf.get("triatt_exact"), dict):
        return dict(fpf["triatt_exact"])
    mod = sys.modules.get("fpf.triatt_exact")
    if mod is None or not hasattr(mod, "report"):
        return {}
    try:
        return dict(mod.report())
    except Exception as e:  # noqa: BLE001
        return {"report_error": repr(e)}


def triatt_exact_evidence(rep: dict | None = None) -> list:
    """The LEVER pairs of triatt_exact: installed, word=exact and bind=tier:exact (the tier word handed to opt_core.kernels.triattn literally), sites (tl |
    tl+pad8: the library names bound), calls (site entries), provider (calls handed to the face), refused (the face's refusals by name, by kind: those calls
    took the site's callable), passthrough (calls outside the provider convention, by reason), kernel (calls the row triattn_exact served) and kernel_refused
    (the member's per-call refusals by reason word: the library op served those), select (the row the word resolves to per probe size on this stack),
    exact_stack (the provider's cc | torch | library vouch key) and opt_core.  ``unit=none`` when the binding is not installed in this process."""
    r = triatt_exact_state(rep)
    if "report_error" in r:
        return [("unit", "report_error"), ("error", _token(str(r["report_error"]))[:80])]
    if not r.get("installed"):
        return [("unit", "none")]
    m = r.get("member") if isinstance(r.get("member"), dict) else {}
    pairs_of = lambda d: ",".join(f"{k}:{n}" for k, n in sorted((d or {}).items())) or "none"
    select = ",".join(f"S{k}:{v}" for k, v in (r.get("select") or {}).items()) or "none"
    return [("installed", 1), ("word", str(r.get("word") or "exact")), ("bind", str(r.get("bind") or "tier:exact")), ("sites", "+".join(r.get("sites") or []) or "none"),
            ("calls", int(r.get("calls") or 0)), ("provider", int(r.get("provider") or 0)), ("refused", _token(pairs_of(r.get("refused")))),
            ("passthrough", _token(pairs_of(r.get("passthrough")))), ("kernel", int(m.get("served") or 0)), ("kernel_refused", _token(pairs_of(m.get("refused")))),
            ("select", _token(select)), ("exact_stack", _token(str(r.get("exact_stack") or "none"))), ("opt_core", _token(str(r.get("opt_core") or "?")))]


def census_rebase(on) -> dict:
    """The census interactions in force in this process: for every lever that is ON and declares ``subsumes`` (registry), the subsumed
    lever's counter and its expected value while the subsuming lever serves that site — ``{subsumed: [(by, counter, expected), …]}``.
    A subsumed lever is still on (its other sites run); its census is re-based on the active lever set, never read as a fallback."""
    from .registry import LEVERS
    out: dict = {}
    for by in LEVERS:
        if by not in on:
            continue
        for lever, counter, expected in LEVERS[by].subsumes:
            out.setdefault(lever, []).append((by, counter, str(expected)))
    return out


def subsumption_pairs(name: str, pairs: list, subsumed: dict) -> list:
    """The evidence pairs a subsumed lever's LEVER line gains: ``subsumed_by=<by>:<counter>(expected=<e>)[,…]`` and, for every such counter
    present among its own evidence pairs, ``<counter>_expected=<e>`` and ``<counter>_census=ok|rebased_mismatch`` (a value other than the
    re-based expectation is NAMED on the line: the reader sees which lever was to serve the site and what ran instead)."""
    rules = subsumed.get(name) or []
    if not rules:
        return []
    have = {k: v for k, v in pairs}
    out = [("subsumed_by", ",".join(f"{by}:{counter}(expected={exp})" for by, counter, exp in rules))]
    for by, counter, exp in rules:
        if counter in have:
            out.append((f"{counter}_expected", exp))
            out.append((f"{counter}_census", "ok" if str(have[counter]) == str(exp) else "rebased_mismatch"))
    return out


def log_lever_lines(rep: dict | None, stream=None) -> list[str]:
    """Print :func:`lever_lines` to stderr (flushed), one line each; returns the lines."""
    return [_core_report.emit(line, stream) for line in lever_lines(rep)]


def partial_refusal_line(rep: dict, when: str) -> str:
    """`NOT ACTIVE: partial activation refused (<when>) — lever: the kit's reason; …` — the exit rule: a mode is all of its levers on
    this card; with one of them unable to run it refuses by name (exit 3) rather than run a subset under the mode's name."""
    why = rep.get("fallback_reasons") or {}
    fb = rep.get("levers_unavailable") or rep.get("levers_fallback") or []
    return (f"{PREFIX} NOT ACTIVE: partial activation refused ({when}) — "
            + "; ".join(f"{n}: {why.get(n) or 'no reason recorded'}" for n in fb)
            + " (a mode is all of its levers on this card: it refuses rather than run a subset under its name)")


def card_lever_set_line(rep: dict | None) -> str | None:
    """`CARD LEVER SET kernel_key=<key>: not in the arm on this card — <lever> (<reason>); …` — the mode's levers that the kernel key's
    compositions row does not carry on this card (stack._classify's NOT_IN_ROW reason): one line, so a per-card difference in the lever
    set is never silent. None when the arm is the full set (nothing to say)."""
    rep = rep or {}
    why = rep.get("fallback_reasons") or {}
    names = [n for n in (rep.get("levers_not_in_arm") or []) if str(why.get(n, "")).startswith((NOT_IN_ROW, STOCK_TRIATT))]
    if not names or not (rep.get("active") or rep.get("dry_run")):
        return None
    return (f"{PREFIX} CARD LEVER SET kernel_key={rep.get('kernel_key') or 'unknown'}: not in the arm on this card — "
            + "; ".join(f"{n} ({why[n]})" for n in names))


def log_activation(rep: dict | None, stream=None) -> str:
    """Print the activation line to stderr unless the report says it was already logged (``rep["logged"]``, set by the print), followed
    by the card lever-set line when the arm on this card is not the full set. Returns the activation line."""
    fresh = not (rep or {}).get("logged")
    line = log_once(rep, activation_line(rep), stream)
    extra = card_lever_set_line(rep) if fresh else None
    if extra:
        print(extra, file=stream or sys.stderr, flush=True)
    return line


def _empty(v) -> bool:
    return v is None or v == [] or v == {} or v == ""


def read_lever_report(path: str | None, pid: int | None = None) -> dict | None:
    """The kit's lever report (``$PTX_LEVER_REPORT``) for one process, merged across its JSON lines: the kit writes several lines per
    pid (clisampler, the trunk record carrying ``applied`` and the counters, stackgraph, fpf_mkpf, fpf_glue_v2) and this package
    appends its own ``protenix_opt`` record last. For every key the LAST line carrying a non-empty value wins, so the trunk's
    ``applied``/``blk_calls``/``deadskip`` survive the later lines that lack them; ``records`` counts the lines merged. With ``pid``
    None every line is merged in file order.

    Returns None when the file is missing, empty, has no parseable line, or (with ``pid``) no line for that pid."""
    if not path or not os.path.exists(path):
        return None
    merged: dict | None = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                if pid is not None and rec.get("pid") != pid:
                    continue
                if merged is None:
                    merged = {"records": 0}
                merged["records"] += 1
                for k, v in rec.items():
                    if not _empty(v) or k not in merged:
                        merged[k] = v
    except OSError:
        return None
    return merged


def count_lines(path: str | None) -> int | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return None


def tally_fields(rec: dict) -> list[str]:
    """``key=value`` fields of the exit tally from one kit report record (only keys that are present)."""
    out = [f"applied={_join(rec.get('applied'))}"]
    ds = rec.get("deadskip")
    if isinstance(ds, dict):
        out.append(f"deadskip={ds.get('n')}/{ds.get('n_expected')}")
    elif ds is not None:
        out.append(f"deadskip={ds}")
    for k in TALLY_KEYS:
        if k in rec:
            v = rec[k]
            out.append(f"{k}=" + (",".join(f"{a}:{b}" for a, b in v.items()) if isinstance(v, dict) else f"{v}"))   # a dict tally (headsplit) as one whitespace-free token
    v02 = rec.get("v02")
    if isinstance(v02, dict) and v02.get("applied"):
        out.append(f"v02={_join(v02.get('applied'))}")
    fpf = rec.get("fpf")
    if fpf:
        out.append(f"fpf_ops={fpf}")
    return out


def exit_tally_line(path: str | None, pid: int | None = None, memory_stats: dict | None = None) -> str:
    """Format the exit tally for this process.

    Source of truth is the report file line for ``pid``; if there is none, ``memory_stats`` (the loaded lever module's ``_STATS``) is
    used and labelled ``source=memory``; if neither exists the line says so (never silent)."""
    pid = os.getpid() if pid is None else pid
    rec = read_lever_report(path, pid=pid)
    if rec is not None:
        return f"{PREFIX} EXIT pid={pid} source=file report={path} " + " ".join(tally_fields(rec))
    n = count_lines(path)
    where = f"report={path or '<unset>'} lines_for_this_pid=0 lines_total={n if n is not None else 'file-missing'}"
    if memory_stats:
        return f"{PREFIX} EXIT pid={pid} source=memory {where} " + " ".join(tally_fields(memory_stats))
    return f"{PREFIX} EXIT pid={pid} no lever counters: levers were never loaded in this process (ptx_trunk2_levers not imported); {where}"


def _memory_stats() -> dict | None:
    lev = sys.modules.get("ptx_trunk2_levers")
    stats = getattr(lev, "_STATS", None) if lev is not None else None
    if isinstance(stats, dict):
        out = dict(stats)
        v02 = sys.modules.get("ptx_fpf_v02")
        v02_stats = getattr(v02, "_STATS", None) if v02 is not None else None
        if isinstance(v02_stats, dict) and "v02" not in out:
            out["v02"] = dict(v02_stats)
        return out
    return None




EXIT_GATE_ENV = "env-route-records"                                  # the env route (PROTENIX_OPT=<mode>, autoload) records the states; the stock process's exit code is its own
_EXIT_GATE = {"value": EXIT_GATE_ENV}


def set_exit_gate(value: str) -> None:
    """Name the exit gate of this process (the CLI verb sets ``cli``; the default is the env route's ``env-route-records``)."""
    _EXIT_GATE["value"] = value


def exit_gate() -> str:
    return _EXIT_GATE["value"]


def _activation_record() -> dict | None:
    """The current activation report (protenix_opt.status()), reconciled with the kit's end-of-run records at exit (stack.reconcile:
    the same states the CLI gates on — partial, the fallbacks), reduced to what the lever report should carry, with
    ``exit_gate`` naming who acts on them (``cli``: the verb exits with them; ``env-route-records``: recorded only)."""
    try:
        from . import stack
        rep = stack.status()
        if rep and rep.get("active") and not rep.get("reconciled"):
            rep = stack.reconcile(rep, stack.late_records(os.environ.get("PTX_LEVER_REPORT"), os.getpid()))
    except Exception:
        return None
    if not rep or not rep.get("mode"):
        return None
    return {"mode": rep.get("mode"), "active": bool(rep.get("active")), "levers_applied": list(rep.get("levers_applied") or []),
            "levers_fallback": list(rep.get("levers_fallback") or []), "levers_not_in_arm": list(rep.get("levers_not_in_arm") or []), "partial": bool(rep.get("partial")),
            "fallback_reasons": dict(rep.get("fallback_reasons") or {}),
            "reconciled": rep.get("reconciled"), "gates": rep.get("gates"), "gate_overrides": rep.get("gate_overrides"),
            "exit_gate": exit_gate(), "package_version": rep.get("package_version"),
            "protenix_version": rep.get("protenix_version"), "gpu": rep.get("gpu"),
            **({"sampler_fuse": rep["sampler_fuse"]} if isinstance(rep.get("sampler_fuse"), dict) else {}),
            **({"apb_levers": rep["apb_levers"]} if isinstance(rep.get("apb_levers"), dict) else {}),
                        **({"triattn_native": native_state()} if "ptx_native_core" in sys.modules else {})}   # the lever's per-process tally (stack.reconcile: sampler_fuse.state; tp.TALLY reads its calls per rank)


def unit_record() -> dict | None:
    """The multi-GPU unit's own statement for this process, when the unit ran here (``ptx_tp.diffusion`` imported): the rank id, the
    diffusion regime it chose (``diffusion.report()["modes"]["sample_diffusion"]``: mode replicated|tp, skip_amp -> fp32|bf16, N, P) and
    its seams ledger — read from the unit's report, never re-derived. None outside the unit (the single-GPU line)."""
    diff = sys.modules.get("ptx_tp.diffusion"); unit = sys.modules.get("ptx_tp")
    if diff is None or unit is None:
        return None
    try:
        sd = (diff.report().get("modes") or {}).get("sample_diffusion")
        out = {"rank": os.environ.get("RANK"), "seams": unit.ledger() if hasattr(unit, "ledger") else None, "diffusion": None}
        if isinstance(sd, dict):
            out["diffusion"] = {"mode": sd.get("mode"), "precision": "fp32" if sd.get("skip_amp") else "bf16", "skip_amp": bool(sd.get("skip_amp")),
                                "N": sd.get("N"), "P": sd.get("P"), "N_sample": sd.get("N_sample"), "N_step": sd.get("N_step")}
        return out
    except Exception as e:  # noqa: BLE001 — the unit's report is the source; a failure to read it is named in the record, never silent
        return {"rank": os.environ.get("RANK"), "error": f"{type(e).__name__}: {e}"[:160]}


def append_activation_record(path: str | None, pid: int | None = None) -> bool:
    """Append one JSON line ``{"pid": ..., "protenix_opt": {...}}`` to the lever report file (the kit's own line stays untouched), so
    the file records which mode the package activated. Returns True when a line was written."""
    rec = _activation_record()
    if not path or rec is None:
        return False
    line = {"pid": os.getpid() if pid is None else pid, "protenix_opt": rec}
    unit = unit_record()
    if unit is not None:
        line["ptx_tp"] = unit
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        return True
    except OSError:
        return False


def exit_line() -> str:
    """The whole EXIT line of this process (the kit's grammar, printed at interpreter exit by the core's tally hook) — and the end-of-run
    bookkeeping beside it: the activation record appended to the lever report; on the env route the reconciled FINAL line first (the CLI
    verb printed its FINAL line itself)."""
    path = os.environ.get("PTX_LEVER_REPORT")
    line = exit_tally_line(path, pid=os.getpid(), memory_stats=_memory_stats()) + f" {ngpu_fields(_activation_record())}"
    append_activation_record(path)
    if exit_gate() == EXIT_GATE_ENV:
        rec = _activation_record()
        if rec and rec.get("active"):
            line = "\n".join([final_line(rec) + f" exit_gate={EXIT_GATE_ENV}"] + lever_lines(rec) + [line])
    return line


def register_exit_tally() -> bool:
    """Register the exit tally atexit hook (idempotent; opt_core.report.register_exit_tally with :func:`exit_line`). Call it BEFORE the
    levers are loaded (see module docstring). Returns True when the hook was registered by this call."""
    return _core_report.register_exit_tally(TAG, exit_line)
