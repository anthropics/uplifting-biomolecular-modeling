"""The tp line's names (``OF3TP_*``: the run variables the launcher ``tp.py`` sets per rank and the row-block / budget size parameters a
caller may set) mapped onto the names ``opt_core.mem.rowpair`` reads (``ROWPAIR_*``, opt_core/mem/rowpair/API.md "Levers read from the
environment"; the core reads no ``OF3TP_*`` name). ``export_core_env()`` writes, for every set ``OF3TP_<k>``, its ``ROWPAIR_<k'>`` into
``os.environ`` before a rank process imports the core's statements (an explicitly set ``ROWPAIR_*`` value wins), and writes the line's
CONSTANTS (``LINE_CONSTANTS``: the placements and policies the tp line runs with — not levers, nothing selects them). ``NOT_CARRIED`` names the
of3tp add-on's switches the tp line has no statement for, each with its reason; they, and a set ``OF3TP_*`` name in no table, are refused by
name (``UnknownSwitch``): a switch that no code reads would otherwise be a silent no-op.
"""
from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional


class UnknownSwitch(ValueError):
    pass


ENV_MAP: Dict[str, str] = {
    # tri-attention bias schedule: the jit schedule's knobs (the gather|jit CHOICE is the kit policy OF3TP_TRIATT_BIAS, KIT_SWITCHES)
    "OF3TP_TRIATT_JIT_QBLOCK": "ROWPAIR_TRIATT_JIT_QBLOCK", "OF3TP_TRIATT_JIT_GROUP": "ROWPAIR_TRIATT_JIT_GROUP", "OF3TP_TRIATT_JIT_PREFETCH": "ROWPAIR_TRIATT_JIT_PREFETCH",
    # run variables (opt_core.mem.rowpair.launch)
    "OF3TP_WORLD": "ROWPAIR_WORLD", "OF3TP_RANK": "ROWPAIR_RANK", "OF3TP_ADDR": "ROWPAIR_ADDR", "OF3TP_PORT": "ROWPAIR_PORT",
    # layout / row blocks / parking (dist, shard, trunk)
    "OF3TP_CHUNK": "ROWPAIR_CHUNK_ALIGN", "OF3TP_ROWBLK_MB": "ROWPAIR_ROWBLK_MB", "OF3TP_RECYCLE_ROWS": "ROWPAIR_RECYCLE_ROWS", "OF3TP_LN_GUARD_ELEMS": "ROWPAIR_LN_GUARD_ELEMS",
    # ring (distributed transposes)
    "OF3TP_TRANSPOSE_BUDGET_GB": "ROWPAIR_TRANSPOSE_BUDGET_GB", "OF3TP_TRANSPOSE_PEERS": "ROWPAIR_TRANSPOSE_PEERS",
    # MSA module + host-resident MSA features (msa, msa_host)
    "OF3TP_OPM_ROWS": "ROWPAIR_OPM_ROWS", "OF3TP_MSA_PAIRAVG_QBLOCK": "ROWPAIR_PWA_QBLOCK", "OF3TP_MSA_PAIRAVG_MAX_GB": "ROWPAIR_PWA_MAX_GB",
    "OF3TP_MSA_TRANS_SHARD": "ROWPAIR_MSA_TRANS_SHARD", "OF3TP_BCAST_CHUNK_GB": "ROWPAIR_BCAST_CHUNK_GB",
    # pair stack (trimul, triatt, pairstack); the APB names are read by this adapter's attention-pair-bias binding
    "OF3TP_TRIMUL_SUB": "ROWPAIR_TRIMUL_SUB", "OF3TP_TRIMUL_GRID": "ROWPAIR_TRIMUL_GRID", "OF3TP_TRIMUL_ROWS": "ROWPAIR_TRIMUL_ROWS",
    "OF3TP_TRIATT_QBLOCK": "ROWPAIR_TRIATT_QBLOCK", "OF3TP_TRIATT_ROWBLOCK": "ROWPAIR_TRIATT_ROWBLOCK", "OF3TP_TRIATT_END_FREE": "ROWPAIR_TRIATT_END_FREE",
    "OF3TP_APB_QBLOCK": "ROWPAIR_APB_QBLOCK", "OF3TP_APB_ROWBLOCK": "ROWPAIR_APB_ROWBLOCK",
    # template embedder
    "OF3TP_TEMPL_NODEDUPE": "ROWPAIR_TEMPL_NODEDUPE",
    # heads / atom-pair statements (heads, frames)
    "OF3TP_CONF_ROWS": "ROWPAIR_CONF_ROWS", "OF3TP_FRAME_ROWS": "ROWPAIR_FRAME_ROWS", "OF3XL_FRAME_BUDGET_MB": "ROWPAIR_FRAME_BUDGET_MB",
    "OF3TP_CLASH_ROWS": "ROWPAIR_CLASH_ROWS", "OF3TP_CLASH_MAX_GB": "ROWPAIR_CLASH_MAX_GB",
    "OF3TP_PARK_STRICT": "ROWPAIR_PARK_STRICT", "OF3TP_PARK_PIN_MAX_GB": "ROWPAIR_PARK_PIN_MAX_GB",
    "OF3TP_POOL_SHRINK": "ROWPAIR_POOL_SHRINK",                 # 0 = keep the A-block row mirror + torch's cached-free pinned blocks through the confidence heads (A/B); default 1
    # diffusion
    "OF3TP_S_WORK_GB": "ROWPAIR_DIFF_WORK_GB", "OF3TP_COND_ROW_BLOCK": "ROWPAIR_DIFF_COND_ROWS", "OF3TP_DIT_Q_CHUNK": "ROWPAIR_DIFF_Q_ROWS",
    "OF3TP_DIT_BIAS_CACHE": "ROWPAIR_DIFF_BIAS_CACHE", "OF3TP_DIT_BIAS_CACHE_GB": "ROWPAIR_DIFF_BIAS_CACHE_GB",
    "OF3TP_ATOM_BAND_W": "ROWPAIR_DIFF_BAND_W", "OF3TP_ATOM_EXTRA_ROWS_MAX": "ROWPAIR_DIFF_BAND_EXTRA_MAX",
    "OF3TP_DIFF_NOISE_SYNC": "ROWPAIR_DIFF_NOISE_SYNC",           # set by the CLI per determinism level (modes.replicated_sync_policy): bcast at det 0, guard at det 1 — not a caller's switch
}
"""``OF3TP_*`` -> the core's name."""

KERNEL_WORDS: Dict[str, str] = {"triatt": "flash_triattn", "trimul": "fpf_v4"}
"""The tp line's triangle kernel words (the tp_triatt / tp_trimul levers): the core's dispatch / provider words ``pairstack`` binds and the launcher's census expects.
``KERNEL_WORDS["triatt"]`` is the word OFF the tier door's compute capabilities (:data:`TRIATT_DOOR_CCS`); on them :func:`triatt_word` names the door."""

TRIATT_DOOR_WORD = "tier:big"
TRIATT_DOOR_CCS = ((9, 0), (8, 0))
"""On the compute capabilities whose ``opt_core.kernels.triattn`` tables carry a ``big`` column (H100 9.0, A100 8.0) the tp_triatt
lever's word is the core's tier door ``tier:big`` (``opt_core.mem.rowpair.triatt.attention_core(kernel="tier:big")``): per row window
the table's memory-tier row (the pre-compiled CUDA triangle-attention row: a 16*S*S-byte fp32 plane instead of flash's ~38*S*S, same
error class), and above the int32 bias bound (H*S*S > 2**31 - 1, S > 23,170 at H = 4) or on any per-call refusal the SAME flash_triattn
q-block path as the word ``flash_triattn`` (the door steps aside by name, counted in the lever's census).  Every other capability
(10.0 / 10.3: no ``big`` column) keeps ``flash_triattn`` by name."""


def triatt_word(cc=None) -> str:
    """The tp_triatt lever's kernel word on a device of compute capability ``cc`` (``(major, minor)``; ``None`` = unknown -> the word by name):
    :data:`TRIATT_DOOR_WORD` on :data:`TRIATT_DOOR_CCS`, ``KERNEL_WORDS["triatt"]`` elsewhere.  The ranks read ``cc`` from their device
    (``pairstack.triatt_kernel``), the launcher from ``nvidia-smi`` (``tp.launch_cc``): one function, one answer per box."""
    try:
        key = (int(cc[0]), int(cc[1])) if cc is not None else None
    except (TypeError, ValueError, IndexError):
        key = None
    return TRIATT_DOOR_WORD if key in TRIATT_DOOR_CCS else KERNEL_WORDS["triatt"]

LINE_CONSTANTS: Dict[str, str] = {
    "ROWPAIR_TEMPL_ORDER": "zparked",          # the z shard is OFF the device (parked) while the template pair stack builds / closes / restores — the template-stage peak drops by ≈ Z/P
    "ROWPAIR_TRIATT_STAGE": "once",            # the tri-attention kernel's bias operand staged ONCE per gathered plane per orientation (triatt.StageSlot armed by triatt_update_ around its row loop; bitwise the per_call outputs; a refusal steps aside BY NAME, census triatt_stage / triatt_stage_aside)
    "ROWPAIR_DIFF_BIAS": "ln_proj",            # the DiT pair-bias PRODUCER = LN + linear_z written head-major straight into the [H, R, N] block (kernels.ln_proj.pair_bias; engine statement by name on refusal, census dit_bias)
    "ROWPAIR_VERBOSE": "1",              # the per-rank census lines the launcher's census reads
    "ROWPAIR_PARK_ZINIT": "recompute",       # no z_init tensor anywhere; a recycle's row block re-runs the init statement (bitwise the parked bytes)
    "ROWPAIR_HOST_SLAB": "lease",            # the u-sized host copies (z parks, tri-mult row mirror) share ONE leased pinned slab per rank
    "ROWPAIR_TRANSPOSE_INPLACE": "1",    # the distributed transposes in place
    "ROWPAIR_MSA_HOST": "rank0",         # raw MSA features on rank 0's host, the selected rows broadcast per recycle
    "ROWPAIR_FREE_ZTRUNK": "1",          # the trunk shard's device storage released for the roll-out
    "ROWPAIR_CONF_PARK_ZTRUNK": "1",     # the trunk shard parked on pinned host through the confidence passes (heads.ZTrunkPlan)
    "ROWPAIR_TEMPL_PARK_Z": "1",         # the z shard parked on pinned host while a real template slot's per-slot pair stacks run (core template.py ParkedStorage; device -u there)
    "ROWPAIR_TEMPL_FEAT_DEVICE": "cuda", # a real template slot's per-row featurizer statements run on the PAIR DEVICE per row block (core template.real_template_rows device=cuda:
                                         #  instead of on the host, where a large templated slot takes minutes per slot group)
}
"""The tp line's constants in the core's names: written by ``export_core_env()`` over any inherited value (placements and policies of the line, not levers)."""

NOT_CARRIED: Dict[str, str] = {
    "OF3TP_TEMPL_REAL": "no real-template switch: on the tp line real template slots are keyed by `--upstream-fix OF3-001` with `--use-templates true` (templ_policy=consume at featurize time: "
                        "tp_rowpair/data.py carries their per-token precursors and each rank computes its rows of the template pair features, template.py), never by a variable; real hits without the fix are refused by name (DataRefused)",
    **{k: "no placement switch: a real slot's per-row featurizer statements run on the host, the stock featurizer's device class (the line's constant ROWPAIR_TEMPL_FEAT_DEVICE=cpu: "
          "bitwise the dense features sliced to rows)" for k in ("OF3TP_TEMPL_REAL_DEVICE", "OF3TP_TEMPL_FEAT_DEVICE")},
    **{k: "no template data-pipeline switch (template id filters, inter-chain template masks): the tp line consumes a templated chain's top-k templates exactly as upstream's sampler returns them "
          "(`--upstream-fix OF3-001`) with the featurizer's own same-chain pair mask"
       for k in ("OF3TP_TEMPL_FIX", "OF3TP_TEMPL_KEEP_IDS", "OF3TP_TEMPL_INTERCHAIN", "OF3TP_TEMPL_INTERCHAIN_SCOPE")},
    "OF3TP_TEMPL_REFUSE_DUMMY": "the all-dummy refusal of a TEMPLATED request; the run's template guard judges that instead (templ_census: TEMPLATES FEATURISED / TEMPLATES DROPPED, exit 5 unless --allow-template-drop) "
                                "and `--upstream-fix OF3-001` consumes real ones",
    "OF3TP_HOST_FEATS": "the of3tp line's host-feature key list; the tp line parks the fixed keys msa, has_deletion, deletion_value (OF3TP_MSA_HOST) and token_bonds (OF3TP_BONDS_HOST)",
    "OF3TP_RESIDENCY_ALL": "the of3tp line's residency-census verbosity; the tp line's census names the CUDA-resident batch tensors (msa.log_batch_residency)",
    "OF3TP_TRIMUL_A_MODE": "the of3tp line's tri-mul A-operand residency word ('full' its only value); the tp line's tri-mul keeps the projected A block of a row sub-block resident (trimul.trimul_update_), the one form",
    "OF3TP_TRIMUL_A_GROUPS_RESIDENT": "the of3tp line's tri-mul A-operand residency count (4 its only value); see OF3TP_TRIMUL_A_MODE",
}
"""``OF3TP_*`` names of the of3tp add-on (``opt/forward/tp``) with no statement on the tp line -> why: refused by name when set (a switch no code
reads would otherwise be a silent no-op)."""


KIT_SWITCHES: Dict[str, str] = {
    "OF3TP_TRIATT_BIAS": "the tri-attention bias schedule POLICY (opt_core's triatt_update_jit_): auto (default: jit — bias sharded [R,N,H] + just-in-time query blocks, "
                         "−(8−8/P)·N² resident bytes per rank — when N > 23,170, where both schedules run flash q-blocks; gather at or below it, where the tier door's pre-compiled CUDA row takes whole rows) | "
                         "gather | jit; read by pairstack.triatt_bias_policy per pair stack, handed to the core's pairstack.bind(triatt_bias=); census triatt_bias_policy=<word>:<bound>@N<n> + the core's triatt_bias=",
    "OF3TP_WRITER_NPZ": "the full-confidence npz container on the tp line's rank-0 writer (model.writer_on_predict_batch_end_patched): stored (default: np.savez — same keys, arrays bitwise "
                        "identical, no zlib pass: an [N,N] PAE at 16K+ deflates for minutes single-threaded) | deflated (upstream's np.savez_compressed); census writer_npz=",
    "OF3TP_DIT_ROWS": "the DiT query-row attention CORE word of the sharded sampler (opt_core mem.rowpair.diffusion.dit_attention_rows): "
                      "big (default: the tier word -> kernels.apb pair_bias_attention_rows serves apb_attn by name, fp32 operands cast to bf16 with census; no [H, q, N] fp32 logits transient) | "
                      "apb_attn | sba[:tf32|ieee|tf32x3] | sdpa | naive (a row by name) | off (today's _attention statement); refusals step aside BY NAME to the statement in q_rows_stock chunks (census dit_rows); read by diffusion.dit_rows_word()",
    "OF3TP_STRUCTURE_FIRST": "structure first: 1 (default) = rank 0 writes each sample's model file through upstream's static writer the moment the sampler returns "
                             "(pLDDT column 0.00 + a _structure_first.json sidecar) and upstream's writer overwrites it after the heads; a structure whose confidence was "
                             "not written is NOT counted by the exit tally (named non-zero exit); 0 = today's order (structures only after the heads); read by "
                             "structure_first_enabled(), never mapped onto a core name",
    "OF3TP_DATA_FORM": "the tp line's data form at P > 1: rank0_bcast (default; rank 0 featurises, the batch reaches every rank through the group) | per_rank "
                       "(every rank featurises); read by tp_rowpair/model.data_form in the ranks and by the launcher (tp.py), never mapped onto a core name",
}
"""``OF3TP_*`` names the kit's own tp statements read (not core levers): known to ``check_known``, absent from ``core_env``."""


def known(name: str) -> bool:
    return name in ENV_MAP or name in KIT_SWITCHES


def check_known(environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Every set ``OF3TP_*`` name of ``environ``; raises ``UnknownSwitch`` naming the ones no table lists."""
    environ = os.environ if environ is None else environ
    names = sorted(k for k in environ if k.startswith("OF3TP_") and (environ.get(k) or "").strip())
    not_carried = [k for k in names if k in NOT_CARRIED]
    if not_carried:
        raise UnknownSwitch("refused: " + "; ".join(f"{k}: {NOT_CARRIED[k]}" for k in not_carried) + " (tp_rowpair/env.py NOT_CARRIED: the tp line has no statement that reads them)")
    unknown = [k for k in names if not known(k)]
    if unknown:
        raise UnknownSwitch(f"refused: unknown tp switch(es) {', '.join(unknown)} (tp_rowpair/env.py ENV_MAP lists the names the tp line reads)")
    return names


def core_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The ``ROWPAIR_*`` assignments implied by the set ``OF3TP_*`` names of ``environ`` (explicit ``ROWPAIR_*`` values in ``environ`` win)."""
    environ = os.environ if environ is None else environ
    out: Dict[str, str] = {}
    for src, dst in ENV_MAP.items():
        v = (environ.get(src) or "").strip()
        if v and not (environ.get(dst) or "").strip():
            out[dst] = v
    return out


def export_core_env(environ=None) -> Dict[str, str]:
    """Write ``core_env()`` and ``LINE_CONSTANTS`` into ``os.environ`` (or the given mutable mapping); returns what was written. Refuses unknown
    ``OF3TP_*`` names."""
    target = os.environ if environ is None else environ
    check_known(target)
    written = {**core_env(target), **LINE_CONSTANTS}
    for k, v in written.items():
        target[k] = v
    return written


def mapping_lines() -> List[str]:
    """``OF3TP_X -> ROWPAIR_Y`` lines (the kit README's switch table)."""
    return [f"{k} -> {v}" for k, v in ENV_MAP.items()] + [f"{k} (the kit's own switch)" for k in KIT_SWITCHES] + [f"{k}={v} (the line's constant)" for k, v in LINE_CONSTANTS.items()]


def structure_first_enabled(environ=None) -> bool:
    """``OF3TP_STRUCTURE_FIRST`` (default 1): the structure-first write of the row-sharded line (opt_core.mem.rowpair.structure_first) is on."""
    return (environ if environ is not None else os.environ).get("OF3TP_STRUCTURE_FIRST", "1").strip() != "0"
