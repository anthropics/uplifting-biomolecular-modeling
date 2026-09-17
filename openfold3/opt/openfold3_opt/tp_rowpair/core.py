"""The ONE import surface of ``opt_core.mem.rowpair`` for this adapter: every binding module reaches the core through ``seam(name)`` /
``fn(name, attr)`` so the core's module layout is named in one table (``SEAMS``) and a core that lacks a module or statement the tp line
binds refuses BY NAME (an ``opt_core.mem.rowpair.RowpairRefused`` naming the module, the statement and the installed opt_core version)
instead of failing with an ImportError / AttributeError deep in a rank. ``register(name, attrs)`` lets a binding module of this package
declare further statements of a seam it binds (one resolver either way); ``census()`` reports, per seam, which bound statements the
installed core has (``seams()``). ``mark(stage)`` is the core's per-stage memory census (``rowpair.census``; off unless
``OPT_CORE_TP_CENSUS=1``).
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, Iterable, Tuple

MIN_CORE = (0, 5, 220, 4)                                    # the opt_core floor this adapter's row statements need (checked at install)
PACKAGE = "opt_core.mem.rowpair"

SEAMS: Dict[str, Tuple[str, ...]] = {
    "dist": ("comm", "Layout", "ctx", "default_ctx", "chunk_align", "row_parts", "world", "env_int", "env_float", "agreed_free_bytes", "checksum", "allreduce_checksum", "require_sharded", "destroy",
             "rank_thread_env", "apply_rank_threads"),
    "launch": ("run_rank_processes", "run_sharded", "rank_env", "free_port", "is_output_rank", "last_run", "RankFailed"),
    "rankdata": ("digest_features", "ranks_agree", "differ_refusal", "broadcast_features", "status_word", "check_data_form", "data_form_word"),
    "shard": ("shard_rows", "unshard_rows", "gather_rows_to", "choose_block_rows", "iter_row_blocks", "produce_rows_", "ln_rows_guarded", "bcast_small",
              "owns_whole_storage", "release_storage_", "regrow_storage_"),
    "ring": ("transpose_shard", "transpose_shard_streamed", "transpose_shard_inplace_", "transpose_band", "ring_pass"),
    "trunk": ("same_rows", "reloffset_rows", "relpos_onehot_rows", "outer_sum_rows", "feature_rows", "pair_mask_rows", "init_pair_shard", "recycle_shard_", "ShardPark",
              "guard_replicated", "guard_rng_replicated", "run_trunk_sharded", "TrunkOut"),
    "template": ("slice_template_inputs_to_rows", "slot_census", "note_all_dummy", "TemplatePairRows", "distogram_rows", "same_or_group_rows",
                 "template_slot_groups", "templ_block_rows", "ParkedStorage", "template_embed_rows"),
    "msa": ("default_opm_rows", "opm_rows_budgeted", "pwa_bias_rows", "default_pwa_qblock", "pwa_rows", "msa_transition_rows", "msa_block_sharded", "msa_module_sharded", "draw_replicated"),
    "msa_host": ("host_mode", "to_host", "HostTensor", "host_placeholder", "park_features", "bcast_chunked_", "rows_to_device", "sync_host_features_", "residency_rows", "residency_text", "is_parked"),
    "transition": ("transition_rows", "opm_rows", "apb_local_queries"),
    "trimul": ("TriMulFns", "trimul_update_", "default_sub", "describe_ablock"),
    "trimul_fused": ("fused_trimul_fns", "emit_line", "describe", "KERNEL_WORDS", "DEFAULT_MIN_TOKENS"),
    "triatt": ("TriAttFns", "triatt_update_", "triangle_bias_rows", "attend_query_blocks", "blocks4", "lever_rows", "attention_core", "emit_core_line", "describe_core", "CORE_KERNELS"),
    "pairstack": ("PairBlockFns", "bind", "transition_update_", "mask_transposed", "pair_block_", "pair_stack_", "env_flag"),
    "heads": ("conf_rows", "logit_rows", "sym_logit_rows", "embed_rows", "pinned_shrink"),
    "structure_first": ("capture", "write_early", "mark_final", "scan", "words", "utc"),
    "confidence": ("ChainIndex", "Context", "ContextReducer", "RowBlockReducer", "expected_value_rows", "gpde_from_full", "max_row_chunk", "per_row_outputs_to_rank0",
                   "gather_single_rows", "AF3_SUMMARIES", "context_reducer_for_layout", "ptm_contexts_from_labels", "conf_reducer_word"),
    "frames": ("frame_rows", "nearest_atoms_rows", "count_pairs_within"),
    "diffusion": ("DiffusionSchedule", "pair_cond_rows", "pair_bias_rows", "PairBiasCache", "DiTBlockFns", "dit_block_sharded", "diffusion_transformer_sharded",
                  "band_plan", "gather_band_rows", "pair_band_rows", "band_lookup", "sync_replicated",
                  "DitBias", "DitKV", "cast16", "dit_rows_core", "dit_attention_rows", "dit_rows_census", "dit_bias_census"),
    "bcast": ("broadcast_tensordict", "tensordict_checksum", "assert_replicated", "sync_tensordict_from_rank0"),
    "census": ("mark", "STAGES", "enabled"),
    "evidence": ("fields", "peak_fields", "gather_peaks", "rows_census", "record_schedule", "schedule", "schedule_fields", "lever_line", "emit_rank0", "rank_failed_line"),
}
"""core module (``opt_core.mem.rowpair.<name>``) -> the statements of it this adapter binds (opt_core/mem/rowpair/API.md "Modules")."""


def register(name: str, attrs: Iterable[str]) -> None:
    """Declare further statements of seam ``name`` a binding module calls (``census()`` then covers them)."""
    cur = SEAMS.get(name, ())
    SEAMS[name] = tuple(cur) + tuple(a for a in attrs if a not in cur)


_MODULES: Dict[str, ModuleType] = {}          # seam -> module (resolved once; the census test injects recording fakes here)


def core_version() -> str:
    try:
        import opt_core
        return str(getattr(opt_core, "__version__", "unknown"))
    except Exception as e:                    # noqa: BLE001 — the refusal names it
        return f"unimportable ({type(e).__name__}: {e})"


def _refused(reason: str):
    try:
        from opt_core.mem.rowpair import RowpairRefused
        return RowpairRefused(reason, lever="n_gpu")
    except Exception:                         # noqa: BLE001 — no core at all: a plain RuntimeError carries the same sentence
        return RuntimeError(reason)


class CoreSeamMissing(RuntimeError):
    pass


def seam(name: str) -> ModuleType:
    """The core module of seam ``name`` (``SEAMS``); refuses by name when the installed opt_core lacks it."""
    mod = _MODULES.get(name)
    if mod is not None:
        return mod
    if name not in SEAMS:
        raise KeyError(f"tp_rowpair.core: unknown seam {name!r} (known: {', '.join(SEAMS)})")
    modname = f"{PACKAGE}.{name}"
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        raise _refused(f"refused: the tp line needs {modname} (opt_core >= {'.'.join(map(str, MIN_CORE))}); installed opt_core "
                       f"{core_version()} lacks it ({type(e).__name__}: {e})") from e
    _MODULES[name] = mod
    return mod


def fn(name: str, attr: str):
    """``getattr(seam(name), attr)`` refused by name when absent (a renamed core statement is a named event, not an AttributeError in a rank)."""
    mod = seam(name)
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise _refused(f"refused: {mod.__name__}.{attr} is absent in opt_core {core_version()} (the tp line binds it: tp_rowpair.core.SEAMS[{name!r}])") from e


def inject(name: str, module) -> None:
    """Serve ``module`` as seam ``name`` (tests: recording fakes; never used by a run)."""
    _MODULES[name] = module


def reset() -> None:
    _MODULES.clear()


def census() -> Dict[str, Dict[str, bool]]:
    """For every seam: which of the bound statements the installed core has (all True on a good install)."""
    out: Dict[str, Dict[str, bool]] = {}
    for name, attrs in SEAMS.items():
        try:
            mod = seam(name)
        except Exception:                     # noqa: BLE001
            out[name] = {a: False for a in attrs}
            continue
        out[name] = {a: hasattr(mod, a) for a in attrs}
    return out


from opt_core.mem.rowpair import census as _census   # CERT-GPU's stage census (TPCENSUS JSONL; no-op unless OPT_CORE_TP_CENSUS=1) — instrumentation only


def mark(stage: str) -> None:
    """``opt_core.mem.rowpair.census.mark(stage)`` (stages trunk_entry/after_template/after_msa/after_pairstack/no_gather/distogram/confidence/
    diffusion_start/diffusion_peak/done); nothing when the installed core has no census module."""
    _census.mark(stage)


def comm():
    """The calling rank's communicator (``opt_core.mem.rowpair.dist.comm()``)."""
    return fn("dist", "comm")()


def layout_of(N: int):
    """The run's row layout for N tokens: ``dist.default_ctx(N)`` (align = ``ROWPAIR_CHUNK_ALIGN`` = the pinned chunk the launcher set)."""
    return fn("dist", "default_ctx")(int(N))


FALLBACK_TAG = "FALLBACK"                                   # the ONE grammar of a named per-call fallback in a rank transcript: `[rowpair rK] FALLBACK <name>=<word>: <why>` (tp.parse_rank_log counts them)


_FALLBACKS: dict = {}                                        # this process's named per-call fallbacks: `<name>=<word>` -> count (fallbacks() reads it for the EXIT line)


def log_fallback(name: str, word: str, why: str) -> None:
    """Name a per-call fallback of this rank (never silent): counted here for this process's EXIT line (report.FALLBACK_SOURCES, label `tp`)
    and written to the transcript, where the launcher's census sums the ranks' lines per name (tp.parse_rank_log / tp.fallbacks)."""
    key = f"{name}={word}"
    _FALLBACKS[key] = _FALLBACKS.get(key, 0) + 1
    comm().log(f"{FALLBACK_TAG} {key}: {why}")


def fallbacks() -> dict:
    """`{<name>=<word>: count}` of this process (a rank) — empty when nothing fell back."""
    return dict(_FALLBACKS)
