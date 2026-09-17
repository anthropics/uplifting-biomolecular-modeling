"""The `--n_gpu P` resource axis of the memory mode: the words and the rules are the core's (opt_core.mem.ngpu, the one producer of the
`n_gpu=P sharding=<scheme>` token and of the refusal sentences); this module holds only what is this engine's: the environment spelling,
the sharding scheme name, the P set the tree serves, and the visible-device reader.

  * `--n_gpu` absent == `--n_gpu 1` == the engine's single-GPU path under any mode: ACTIVE / EXIT carry `n_gpu=1 sharding=none`.
  * `--n_gpu P>1` under `exact` / `fast` / `off` is refused by name (NG.REFUSE_MODE); with fewer than P visible devices refused by name
    (`refused: n_gpu=P visible=K`); a P outside SUPPORTED is refused by name (REFUSE_UNSERVED). Every refusal exits 3; nothing shrinks P.
  * environment route: `PROTENIX_V1_OPT_N_GPU=P` is the .pth hook's spelling of `--n_gpu P` (stack.activate reads it when the verb did not
    pass one).
  * `--n_gpu P>1` exports the row-sharded pair stack's host-parking levers by their core names (`large_input_regime`, TP_EXPORTS)
    in the launching process and in every rank process, unless the caller set a name; the P=1-only memory levers the sharded path
    replaces are off by the line's in-process selection in every rank (big.rowpair_switches).
"""
from __future__ import annotations

import os

from typing import Optional

from opt_core.mem import ngpu as NG            # the token / refusal producer (opt_core.mem.ngpu)

ENV_N_GPU = "PROTENIX_V1_OPT_N_GPU"
SCHEME = "rowpair"                              # the pair representation row-sharded over P devices (opt_core.mem.rowpair)
SUPPORTED = (1, 2, 4, 8)                        # the P set `--mode big` serves in this tree; a P outside it is refused by name
REFUSE_UNSERVED = "refused: n_gpu={P} not served: this tree's big serves n_gpu in {S}"


def from_environ(environ=None) -> Optional[str]:
    import os
    environ = os.environ if environ is None else environ
    v = (environ.get(ENV_N_GPU) or "").strip()
    return v or None


def effective(cli_value=None, environ=None) -> int:
    """The run's P: `--n_gpu` when given, else PROTENIX_V1_OPT_N_GPU, else 1 — a positive integer or ValueError (usage)."""
    v = cli_value if cli_value not in (None, "") else from_environ(environ)
    return NG.check_n_gpu(1 if v is None else v)


def visible_devices() -> int:
    """The device count torch reports (0 without torch or without CUDA); never used to size P, only to refuse."""
    try:
        import torch
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


REFUSE_WORLD = "refused: n_gpu={P} inside a launch of world {W}"


def launch_world() -> int:
    """The world size of the launch this process is a rank of (opt_core.mem.rowpair.launch's rank environment); 1 outside a launch."""
    from opt_core.mem.rowpair import launch
    return int(launch.world_size())


def admit(n_gpu, mode: str, visible: Optional[int] = None) -> int:
    """Every rule in order — the mode rule, the visible-device rule (P>1, in the launching process; a rank process of a launch sees its one
    device and must name the launch's own P), the served-P rule; returns P or raises NG.NGpuRefused (`.reason` = the sentence the verb prints
    after `NOT ACTIVE:`)."""
    p = NG.refuse_unless_big(n_gpu, mode)
    if p > 1:
        w = launch_world()
        if w > 1:
            if p != w:
                raise NG.NGpuRefused(REFUSE_WORLD.format(P=p, W=w))
        else:
            NG.refuse_unless_visible(p, visible_devices() if visible is None else visible)
    if p not in SUPPORTED:
        raise NG.NGpuRefused(REFUSE_UNSERVED.format(P=p, S=set(SUPPORTED)))
    return p


def fields(n_gpu) -> str:
    """`n_gpu=P sharding=rowpair` / `n_gpu=1 sharding=none` — the core's rendering, verbatim."""
    return NG.active_fields(n_gpu, SCHEME)


def record(n_gpu) -> dict:
    """The activation report's fields of the axis."""
    p = NG.check_n_gpu(n_gpu)
    return {"n_gpu": p, "sharding": NG.sharding_value(p, SCHEME)}


def mismatch_reason(requested: int, active) -> str:
    """`reason=n_gpu_mismatch requested=P active=Q` — the fail-closed word when the P a run reports (a rank's ACTIVE line, the group's world,
    the launcher's rank count) is not the P the command line requested: never a pass on fewer devices than asked."""
    return f"reason=n_gpu_mismatch requested={int(requested)} active={active}"


def assert_ranks_report(requested: int, rank_logs) -> None:
    """After a P>1 launch: every rank's transcript carries `ACTIVE … n_gpu=P sharding=rowpair` with P == requested and there are exactly P
    transcripts; else NGpuRefused(mismatch_reason). `rank_logs` = the P log paths in rank order."""
    logs = list(rank_logs)
    if len(logs) != int(requested):
        raise NG.NGpuRefused(mismatch_reason(requested, f"{len(logs)}_ranks"))
    want = fields(requested)
    for r, path in enumerate(logs):
        seen = None
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if " ACTIVE " in line and "n_gpu=" in line:
                        seen = line
                        break
        except OSError:
            seen = None
        if seen is None or f" {want}" not in seen:
            got = "no_active_line" if seen is None else (seen.split("n_gpu=", 1)[1].split()[0] if "n_gpu=" in seen else "?")
            raise NG.NGpuRefused(mismatch_reason(requested, f"rank{r}:{got}"))


SUPERSEDED_LEVERS = ("relp_lean", "recycle_carry", "diffusion_cond_chunk", "conf_head_chunk")   # P=1 levers whose sites the row-sharded path REPLACES (tp.py docstring): off by name under n_gpu>1 — the line's in-process selection in every rank (big.rowpair_switches; LEVER line `state=off reason=flag`)


TP_EXPORTS = {                                  # the ×P line's exports: the core's host-parking levers of the row-sharded pair stack (opt_core/mem/rowpair/API.md
    "ROWPAIR_PARK_ZINIT": "recompute",          # "Levers read from the environment"), stated in every rank process unless the caller set the name (large_input_regime):
    "ROWPAIR_MSA_HOST": "rank0",                #
    "ROWPAIR_CONF_PARK_ZTRUNK": "1",            #
    "ROWPAIR_FREE_ZTRUNK": "1",                 #
    "ROWPAIR_TRANSPOSE_INPLACE": "1",           #
    "ROWPAIR_RANK_THREADS": "auto",             #
    "ROWPAIR_HOST_SLAB": "lease",               #
    "ROWPAIR_TRIATT_STAGE": "once",             #
}                                               # ROWPAIR_PARK_ZINIT=recompute — no copy of the z_init shard [R, N, c_z] is kept between recycles: a recycle's row block re-runs the init
                                                #   statements on the init grid (ShardPark.recompute; bitwise equal to a parked copy; census zinit_park=recompute park_z_init=recompute
                                                #   zinit_recompute_calls/_s). `1` instead PARKS the shard on pinned host (opt_core.mem.rowpair.trunk.ShardPark: one D2H copy, the device
                                                #   storage released, row blocks staged back per recycle; the device holds the z shard + one row block instead of two shards through the
                                                #   trunk; census `park_z_init=host_pinned|host_pageable:<kind>` + `park_z_init_gib` on the rowpair LEVER line).
                                                # ROWPAIR_RANK_THREADS=auto — each rank's CPU threads capped to cores // ranks by the core launcher (census rank_threads=).
                                                # ROWPAIR_HOST_SLAB=lease — the family's u-sized pinned host copies (the trunk shard's confidence park, the tri-mult row mirror) share ONE
                                                #   leased slab per rank (census host_slab=lease host_slab_gib= host_slab_holder=<single token>).
                                                # ROWPAIR_TRIATT_STAGE=once — the triangle-attention kernel's bias operand is staged ONCE per gathered plane per orientation by the core's
                                                #   triatt_update_, every row window launching the kernel alone — bitwise equal to per-call staging (census triatt_stage=once:<row>
                                                #   triatt_stage_planes= triatt_stage_gib=); `per_call` stages it per row window.
                                                # ROWPAIR_MSA_HOST=rank0 — the raw MSA features [S_msa<=16384, N] (msa int64 + has_deletion + deletion_value fp32: 16 B per MSA row and
                                                #   token) never reach a device whole and are held by rank 0 only (pinned host; opt_core.mem.rowpair.msa_host): the per-cycle subsample's
                                                #   k <= S_msa rows are gathered on rank 0's host and broadcast to every rank's device (tp.msa_module_tp); `all` = every rank keeps a host
                                                #   copy and gathers its own; census msa_host_mode= msa_host_pinned_gib= msa_host_rows=<mode>:<rows shape>:<GiB>:<s> msa_raw=host:<mode>
                                                #   msa_sample_rows=<k>.
                                                # ROWPAIR_CONF_PARK_ZTRUNK=1 + ROWPAIR_FREE_ZTRUNK=1 — the TRUNK shard after the trunk (opt_core.mem.rowpair.heads.ZTrunkPlan, the
                                                #   core's one statement of which mechanism when; tp.rollout_entry / tp.confidence_tp bind it): at roll-out entry the distogram rows
                                                #   are taken from the device shard, then the shard is PARKED on pinned host and its device storage released before the conditioned
                                                #   pair rows are allocated; the roll-out, the distogram and every confidence pass run with one pair shard less on the device and each
                                                #   confidence pass builds its pair input from host-served rows (device: ONE shard-equivalent per pass at any N_sample, vs the trunk
                                                #   shard + the pass's pair input resident). FREE alone (park off): a single same-dtype last-use pass embeds IN PLACE into the shard's
                                                #   storage; census conf_ztrunk_entry= conf_ztrunk=<word per pass> conf_ztrunk_host_gib= conf_pairstack=inplace zcond_src=parked.
                                                # ROWPAIR_TRANSPOSE_INPLACE=1 — every pair block's ending-node triangle attention runs on TRANSPOSED shards; the distributed
                                                #   transposes (opt_core.mem.rowpair.ring, read by pairstack.pair_block_: the trunk Pairformer, the MSA module's and the template
                                                #   embedder's pair blocks, the confidence stack all go through it) swap row blocks pairwise into the shard's OWN storage both ways
                                                #   instead of into a second shard-sized buffer: −u/P transient in every pair block; numerics-neutral (bit-equal); census transpose_form=inplace:<k>.


def regime_names() -> tuple:
    """Every environment name `large_input_regime` states (TP_EXPORTS) — what a P>1 launch writes into the rank processes' environment."""
    return tuple(TP_EXPORTS)


def large_input_regime(environ, n_gpu: int = 2) -> dict:
    """Under a P>1 launch the row-sharded pair stack's host-parking levers are exported (TP_EXPORTS: the core's `ROWPAIR_*` names at the
    line's values, source `n_gpu>1:tp_exports`) unless the caller set them (a caller's value, `0` included, wins and is recorded as
    `caller`). The four single-GPU levers the sharded path replaces (SUPERSEDED_LEVERS) are off by the line's own in-process selection in
    every rank (big.rowpair_switches), not through the environment; the sampler CUDA graphs and the DiT hoist are off under the mode at
    every P (big.BASE_LEVERS_OFF), so the line's expandable-segments allocator — which the sharded path's varying collective buffers need
    against fragmentation — engages in every rank. Returns {name: (value, source)}. No size keys anything here."""
    out = {}
    for name, value in TP_EXPORTS.items():
        val = (environ.get(name) or "").strip()
        if val:
            out[name] = (val, "caller")
        else:
            environ[name] = value
            out[name] = (value, "n_gpu>1:tp_exports")
    return out




