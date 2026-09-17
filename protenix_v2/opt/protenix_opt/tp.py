"""The multi-GPU LINE of ``big``: tensor-parallel Protenix inference (the carried unit ``opt/forward/PTX_TP``, package ``ptx_tp``)
— the ``--n_gpu P`` resource axis of the memory mode for P > 1 (``pred --mode big --n_gpu P``), composing on the kit's ``fast`` levers.
The interface is the shared core's (``opt_core.mem.ngpu``): ``--n_gpu`` is explicit, default 1 (the mode's single-GPU line, ``lowmem``);
P > 1 under ``exact`` / ``fast`` / ``off`` is refused by name (sharded reductions reorder sums: never bitwise); fewer than P visible GPUs is
refused by name (``refused: n_gpu=P visible=K``; never shrunk); an accepted run's ACTIVE / FINAL / EXIT lines carry ``n_gpu=P
sharding=rowpair`` (``active_fields``).

How the composition works: the launcher of the unit (``python -m ptx_tp.launch --nproc N``) starts N ranks under ``torchrun`` and every
rank runs the stock CLI entry with the unit's hooks; each rank interpreter carries ``PROTENIX_OPT=fast`` so the installed
``protenix_opt_autoload.pth`` activates the fast levers in it at start (the ENV route, `README.md`), and the unit's hooks
(``ptx_tp:apply_from_env``, the lazy relative-position rows ``PTX_TP_RELP=lazy``, the size-guard lift ``XL_LIFT_GUARD=1``) install at the
entry. The line passes ``--lazy-relp --lift-guard`` and never ``--dap`` (``LINE_ARGS``); the launcher itself appends
``--trimul_kernel torch`` to the stock arguments (its sharded TriMul is the torch statement) — that flag is the line's one stock-argument
delta and the manifest records it.

Memory-first at every size: the line forces every size / threshold gate of the unit to its sharded value in every rank (``TP_GATES``:
the diffusion pair conditioning + DiffusionTransformer, the atom encoder's pair band, the confidence / summary seams and the MSA / template
seams run their row-sharded statements at every N; no Mode-S path switches to a replicated statement below a token count); a caller
environment that opens a gate is refused by name (``gate_refusal``). The ACTIVE line states the gates and the schedule census
(``CENSUS``: the row-sharded pair tensors and the tensors replicated in every rank by design).

Fail-loud by construction: ``--n_gpu P`` needs P visible GPUs and the unit fully present (``preflight``); after the run the ranks'
phase logs and the launcher's rank banners must show world size N, every rank alive to its last phase, the collective backend, the
row-shard map, every effective seam ``tp`` and the diffusion regime ``tp`` (``events``, ``reconcile_ranks``); anything short is NOT
ACTIVE / FAIL by name — a run that fell back to fewer GPUs or to a replicated statement is never exit 0."""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from opt_core.mem import ngpu as _ngpu                  # the one producer of the n_gpu token text and refusal sentences (shared core)
from opt_core.mem.rowpair import RowpairRefused          # the core's by-name refusal of a layout that cannot shard
from opt_core.mem.rowpair import rankdata as _rankdata   # the one rule for the rank interpreters' str-hash seed (PYTHONHASHSEED) and its census word (shared core)

from . import kits
from . import tp_route

LINE_MODE = "big"                                     # the mode this line belongs to
BASE_MODE = "fast"                                      # the composition base: every rank activates this mode's levers
LINE = "tp"                                             # the line's name in the ACTIVE line and the manifest
SINGLE_LINE = "single"                                  # the mode's other line: its single-GPU composition (modes.py big), activated in-process
ENV_NGPU = "PROTENIX_OPT_N_GPU"                         # the GPU count, env form (the env route cannot launch ranks: a value > 1 there is refused by name, lowmem.apply)
FLAG_NGPU = "--n_gpu"                                   # the GPU count, CLI form: explicit, default 1 (opt_core.mem.ngpu: never auto-detected, never shrunk)
DEFAULT_NGPU = 1                                        # absent flag == --n_gpu 1: the mode's single-GPU line; P > 1 selects this line (big only)
SCHEME = "rowpair"                                      # the sharding scheme this line drives (opt_core.mem.ngpu.SCHEMES): the pair representation row-sharded over P ranks
IMPL = "ptx_tp"                                         # the implementation named on the ACTIVE line (impl=): the carried PTX_TP unit's engine seams;
                                                        # its engine-free layers route to the core in every rank (tp_route.ROUTES; routed= on the line)
LAUNCHER = "torchrun_loopback"                          # launcher= on the line: the unit's launcher (LAUNCHER_MODULE) execs torchrun with an explicit c10d rendezvous
                                                        # on 127.0.0.1:<free port> (protenix_opt.tp_bind.launch, armed by tp_route.install) — no hostname resolution
DET_GUARDS = "PTX_DET"                                  # the unit's replicated-tensor checksum guards (ptx_tp.det_recipe: the received features vs rank 0's,
                                                        # the recomputed replicated activations per recycle, the diffusion noise / noisy coordinates per step +
                                                        # the final coordinates) run under the det recipe only: deterministic kernels make them rank-identical, a
                                                        # mismatch is a chimera and is refused by name; outside the recipe fast's kernels recompute the activations
                                                        # per rank at run-to-run reproducibility, not bitwise, so the checksums are not a statement there
STRATEGY = "F7.tensor_parallel"                         # the canonical strategy id of the row-sharded pair stack (opt_core/STRATEGIES.json)
UNIT = "PTX_TP"                                         # opt/forward/PTX_TP (kits.KITS)
UNIT_PKG_DIR = "PTX_TP_ADDON"                           # the directory inside the unit that holds the ``ptx_tp`` package
LINE_ARGS: Tuple[str, ...] = ("--lazy-relp", "--lift-guard")   # the runnable line: no --dap; PTX_TP_RELP=lazy; XL_LIFT_GUARD=1
LAUNCHER_MODULE = "ptx_tp.launch"
STOCK_ARG_DELTA: Tuple[str, ...] = ("--trimul_kernel", "torch")   # appended by the launcher on every TP run (ptx_tp/launch.py)
PHASE_LOG_DIR = "tp"                                    # <records dir>/tp/: phases_rank<r>.jsonl, launch.log, runmeta_<label>.json (cli.records_dir: never the stock output directory)
TP_DROPPED: Tuple[str, ...] = ("stackgraph", "sampler_graph", "sampler_graph_cache_policy", "sampler_prep", "sampler_reach", "sampler_admit", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused")   # fast's graph levers, not part of the line: the unit's trunk
# cond_dedupe / dit_fused / dit_lowp / atom_fused (the sampler's fused-stack levers): not part of the multi-GPU line either — its row-sharded diffusion binding replaces the token blocks' functions the fused stack would own; nobody has composed the two (left uninstalled by TP_PRE below, named on the line's record as dropped, never refused).
                                                        # replaces the graphed pair stack and the sampler graph's DiT hoist refuses under TP by its own rule
TP_PRE: Dict[str, str] = {"PTX_BLK_GRAPH": "0", "PTX_SAMPLER_GRAPH": "0", "PTX_COND_DEDUPE": "0", "PTX_DIT_FAST": "0", "PTX_DIT_LOWP": "off", "PTX_ATOM_FAST": "0"}   # the switches that leave them uninstalled in every rank (env.sh: a caller value wins)
# The unit's size / threshold gates, FORCED to the sharded path in every rank at every N (memory-first semantics of the mode: no Mode-S path
# of the unit switches to a replicated statement at any size under this line). Each value below is the unit's "always sharded" setting; the
# unit's own defaults open them below a token count (ptx_tp/launch.py build() setdefault 3841 for the two diffusion gates). A caller
# environment that carries another value is refused by name (``gate_refusal``): a size-gated (speed-oriented) tensor-parallel behaviour
# is not a setting of this line. ``events`` re-reads the ranks' seams records and ``reconcile_ranks`` the diffusion regime: a seam record
# not readable as ``tp`` or a ``replicated`` diffusion regime after the run is a named failure.
TP_GATES: Dict[str, str] = {
    "PTX_TP_DIFF_REPLICATE_BELOW": "0",                 # ptx_tp/diffusion.py:972 — open: N < value runs the whole diffusion module (pair conditioning, the 24-block
                                                        # DiffusionTransformer's pair bias, atom encoder/decoder) as the STOCK statements on an all-gathered fp32
                                                        # pair_z [N, N, c_z] in every rank; forced 0: z_cond rows once per rollout + local query rows per block
    "PTX_TP_F2_GATHER_BELOW": "0",                      # ptx_tp/diffusion.py:410 — open: N < value computes the atom encoder's pair band by the stock statement on a
                                                        # transiently all-gathered pair_z; forced 0: the band is gathered from the pair ROWS (2W+1 window)
    "PTX_TP_DROPOUT_FULL_MAX": "0",                     # ptx_tp/trunk.py:225,580 — open: N <= value all-gathers the recycling projection [N, N, c_z] to draw torch's
                                                        # full-tensor mc-dropout mask (items whose mc-dropout draw is on); forced 0: the P-invariant block-seeded
                                                        # row-local mask at every N (tier-2-by-mask against one GPU's Philox draw — inside the mode's tier-2 word)
    "PTX_TP_DIFF_ATTN": "rowsplit",                     # ptx_tp/diffusion.py:144,577 — replicated: every DiffusionTransformer block all-gathers its full [1, H, N, N]
                                                        # pair bias; forced rowsplit: queries = this rank's rows, bias rows local, the output rows all-gathered
    "PTX_TP_APB_SHARD_ABOVE": "0",                      # ptx_tp/pairformer.py:141 — open: N < value runs the Pairformer's single attention with replicated queries on
                                                        # an all-gathered [N, N, 16] pair bias per block; forced 0: local query rows on their own bias rows at every N
    "PTX_TP_DROP_BOND_MASK": "1",                       # ptx_tp/runner_hooks.py install_bond_mask_drop — the featurizer's int64 [N_atom, N_atom] bond mask dropped in
                                                        # every rank (the mode's drop_bond_mask lever per rank; p1_levers= says per_rank): a caller 0 is refused by name
}
# Size-named settings of the unit that are NOT gates of this line (no pair-shaped tensor becomes whole on a rank on either side of them): the
# offload / allocator / recompute thresholds PTX_TP_HOST_FEATS_ABOVE (input features stay on the host), PTX_TP_GC_ABOVE_N (allocator garbage
# collection cadence), PTX_TP_TEMPL_PARK_ABOVE_GB and PTX_TP_ZINIT_RECOMPUTE_ABOVE_GB (park / recompute a shard by its size in GB) keep the unit's
# values.
NOT_GATES: Tuple[str, ...] = ("PTX_TP_HOST_FEATS_ABOVE", "PTX_TP_GC_ABOVE_N", "PTX_TP_TEMPL_PARK_ABOVE_GB", "PTX_TP_ZINIT_RECOMPUTE_ABOVE_GB")
# big's single-GPU memory levers (big.LINE) under the multi-GPU line, per lever: applied in every rank, or replaced by the row-sharded
# statement that makes it moot (named, never dropped silently; p1_levers= on the ACTIVE line, tp.p1_levers in the manifest).
P1_LEVERS_UNDER_TP: Dict[str, str] = {
    "drop_bond_mask": "per_rank",                            # the unit installs the featurizer's bond-mask drop in every rank (ptx_tp/runner_hooks.py install_bond_mask_drop)
    "cond_chunk": "replaced_by_rowpair:z_cond_rows",         # DiffusionConditioning's pair path runs on this rank's rows once per rollout (PTX_TP_DIFF_REPLICATE_BELOW=0)
    "apb_bias_chunk": "replaced_by_rowpair:bias_rows",       # attention-pair-bias reads only this rank's pair rows (DiT rows-only bias cache; PTX_TP_APB_SHARD_ABOVE=0)
    "cache_release": "unit_release_points",                  # allocator releases at the unit's own seams (ptx_tp/pairformer.py release_device_caches, trunk.py _release_cached)
    "relp_lazy": "replaced_by_rowpair:relp_rows",            # relative-position one-hot rows generated per row block of the shard, lazily (--lazy-relp: PTX_TP_RELP=lazy; no plane on any rank)
    "msa_zfree": "replaced_by_rowpair:msa_rows",             # the MSA module's pair path is row-sharded per rank (no whole block-input z on a rank to release)
    "diffcache_free": "replaced_by_rowpair:z_cond_rows",     # the conditioning pair cache is this rank's rows (the bound diffusion seam owns its residency)
}
BIND_ENV: Dict[str, str] = {
    "ROWPAIR_DIFF_BIAS_CACHE_GB": "8",                  # the DiffusionTransformer rows-only pair-bias cache cap per rank under the bound diffusion seam
}                                                       # (the core rule's budget; the seam's own decision — tp_bind.diffusion.bias_cache_decision, policy
                                                        # ``fit`` — passes an explicit decision and this budget applies only under ROWPAIR_DIFF_BIAS_CACHE_POLICY=core)
RANK_ENV_DEFAULTS: Dict[str, str] = {
    "ROWPAIR_RANK_THREADS": "auto",                     # the core's per-rank CPU-thread cap (opt_core.mem.rowpair.dist.apply_rank_threads at rank entry,
}                                                       # tp_route.install): cores // ranks; a caller's value (inherit | <n>) wins; census rank_threads=<n>
# The replicated-tensor sync policy of the bound diffusion seam BY DET LEVEL (core word ROWPAIR_DIFF_NOISE_SYNC; printed as noise_sync= on the
# ACTIVE line, recorded in the manifest): det 0 -> bcast (rank 0 is authoritative at every sync point — the per-denoiser-call state x_noisy
# after augmentation + noise; the loop's replicated statements are not bitwise run to run without --det, so a guard would turn benign last-bit
# drift into a refusal); det 1 -> guard (strict: deterministic kernels make every rank's replicated tensors bitwise equal, and a mismatch is a
# real defect refused by name). The sampler-exit adoption of rank 0's coordinates and the diff_rank_spread_A census / refusal run at both levels.
NOISE_SYNC_BY_DET: Dict[int, str] = {0: "bcast", 1: "guard"}
NOISE_SYNC_ENV = "ROWPAIR_DIFF_NOISE_SYNC"


def noise_sync(det_level: Optional[int]) -> str:
    return NOISE_SYNC_BY_DET[1 if det_level else 0]
ACROSS_P = "tier2(gemm_m=rows_per_rank)"                # outputs across GPU counts under --det: the per-element arithmetic is the dense statement's, the GEMM row
                                                        # extent M is the rank's row count R, and kernel selection follows the shape — byte-identical across P where
                                                        # the shapes select the same kernels, tier 2 otherwise; never claimed exact. The engine's in-place tri-mult chunk
                                                        # is named beside it (INPLACE_CHUNK).
INPLACE_CHUNK = 256                                     # protenix triangle_multiplicative _inplace_chunk_size at inference (the stock k-grid the sharded TriMul keeps)
# The schedule census of the line (printed on the ACTIVE line, recorded in the manifest): what is row-sharded over the P ranks and what is
# replicated in every rank BY DESIGN (O(N) / O(N_atom) / O(S_msa x N / P) tensors) — the unit's statements (ptx_tp docstrings: trunk.py,
# msa.py, template.py, diffusion.py, confidence.py); nothing [N, N, c] is whole on a rank.
CENSUS: Dict[str, Tuple[str, ...]] = {
    "sharded_rows": ("z_init", "relpos_onehot(lazy rows)", "z(trunk, 48 pairformer blocks)", "z_template(pair stack)", "z_msa(outer product mean rows, pair-weighted averaging rows, pair stack)",
                     "m(msa rep: token-sharded)", "distogram logits(row blocks, reduced)", "z_confidence(4 pairformer blocks)", "pae/pde logits(row blocks, reduced on the fly)",
                     "z_cond(diffusion pair conditioning rows, once per rollout)", "diffusion pair bias(rows-only cache / per block)", "atom-encoder pair band(from pair rows)"),
    "replicated_by_design": ("input features", "s_inputs [N, 449]", "s [N, 384] (single rep; attention-pair-bias queries = local rows, s rows gathered per block)",
                             "msa raw features (host)", "atom features / coordinates [N_atom, ...]", "diffusion noise (rank-identical RNG, PTX_TP_RNG_SYNC=1)",
                             "token single conditioning", "plddt / resolved logits [N, ...]"),
    "gathers_inside_trunk": ("triangle-attention starting-node bias [N, N, 4 heads] all-gathered in row blocks (the small bias, never z)",
                             "s rows after attention-pair-bias per block"),
}
# What proves a base-mode lever EXECUTED in a rank, from the kit's own lever report (record key, counter paths summed): under TP the unit's
# tp_pairformer_block runs its own TriMul / tri-attention / transition over the stock block's parameters, so the pair-stack levers are
# installed but never reached — a lever with a zero tally is `inert_under_tp`, by count; a lever with no counter is an environment lever.
TALLY: Dict[str, Tuple[str, Tuple[Tuple[str, ...], ...]]] = {
    "trimul_core": ("trimul_routes", (("lever_calls",),)), "trimul_core_exact": ("trimul_routes", (("lever_calls",),)),   # src/ptx_trimul_routes.py census (the trunk stacks run on every rank)
    "k2b_flash_triattention": ("blk_att_served", (("k2b",),)),
    "blk2_block_path": ("blk_calls", ()),
    "blk2_chunked_k2b": ("blk_calls_chunked", ()),
    "mk_pf": ("mkpf", (("f1_start",), ("f1_end",), ("f1_padded",), ("f3",))),
    "glue_v2": ("glue_v2", (("prologue_v4",), ("prologue_v4_padded",), ("epilogue_v3",), ("transition_ws",))),
    "t1_fused_transition": ("t1_fused_calls", ()),
    "deadskip": ("deadskip", (("n",),)),
    "stackgraph": ("stackgraph", (("captures",), ("replays",), ("eager",))),
    "sampler_graph": ("clisampler", (("sampler", "captures"), ("sampler", "replays"))),
    "triattn_native": ("protenix_opt", (("triattn_native", "counts", "native"), ("triattn_native", "counts", "other_row"))),   # provider-served calls of the block core's attention slot (native + the other rows the tier word names)
    "transition_core_exact": ("fpf", (("transition_core", "served"),)), "transition_core": ("fpf", (("transition_core", "served"),)),   # the provider binding: the trunk runs on every rank
    "triatt_prologue_cuda": ("fpf", (("procuda", "cuda"),)),                                # prologue node calls: the trunk stacks run on every rank
    "pf_attn": ("protenix_opt", (("apb_levers", "pf_attn", "calls"),)),                    # the line's row-sharded Pairformer block functions do not call the instance method: inert by count there
    "atom_attn_exact": ("apb_atom_exact", (("calls", "kernel"),)),                    # the exact composition only (no TP mode lists it); replicated atom attention if it ever did
    "opm_fused": ("protenix_opt", (("apb_levers", "opm_fused", "calls"),)), "pwa_fused": ("protenix_opt", (("apb_levers", "pwa_fused", "calls"),)),   # the MSA module runs replicated on every rank                          # kernel calls: the trunk stacks run on every rank (row-sharded)                     # per-head kernel calls: the trunk stacks run on every rank (row-sharded), the tally scales with the ranks
    "dit_attn_exact": ("fpf", (("dit_attn_exact", "calls"),)),
    "triatt_exact": ("fpf", (("triatt_exact", "calls"),)),                            # the exact composition only (no mode of the line lists it); rides the trunk record: the trunk stacks run on every rank if it ever did
    "dit_attn": ("protenix_opt", (("apb_levers", "dit_attn", "calls"),)),            # the line's row-sharded DiT block functions do not call Attention.forward: 0 under TP by construction (inert by count)
    "dit_attn_fp16": ("protenix_opt", (("apb_levers", "dit_attn", "calls"),)),       # the precision lever rides dit_attn's kernel calls: inert by count under the line with it
    "atom_attn": ("protenix_opt", (("apb_levers", "atom_attn", "calls"),)),          # the atom attention is replicated under the line: served on every rank
    "sampler_fuse": ("protenix_opt", tuple(("sampler_fuse", "calls", k) for k in ("ada", "gate", "res", "swiglu")) + tuple(("sampler_fuse", "regime_stock_calls", k) for k in ("ada", "gate", "block")) + tuple(("sampler_fuse", "offsig_calls", k) for k in ("ada", "gate", "res", "swiglu"))),   # the package record's per-process fused-kernel calls and regime-dispatch stock-expression calls (sampler_fuse.state)
}
SEAMS = ("pairformer", "msa", "diffusion", "confidence", "summary", "trimul", "triatt", "rowlocal", "trunk")
APPLIED_RE = re.compile(r"\[ptx_tp r(\d+)\] APPLIED P=(\d+) feat=(\S+) relp=(\S+) zinit_recompute=(\S+) extras=(\[[^\]]*\]) seams: ([^\n]*)")
SEAM_RE = re.compile(r"\b(" + "|".join(SEAMS) + r")=(tp)")          # the closed vocabulary (<seam>=tp): another writer's bytes glued to the line cannot change a value
REPLICATED_RE = re.compile(r"\[ptx_tp r0\] N_token=(\d+) < P\*B=(\d+) \(or labels given\): replicated regime -> STOCK main loop on every rank")
ZERO_ROWS_RE = re.compile(r"\[ptx_tp r0\] WARNING layout B=(\d+): ranks (\[[^\]]*\]) own ZERO rows at N=(\d+), P=(\d+)")
EFFECTIVE_RE = re.compile(r"per-rank peak alloc GiB = (\[[^\]]*\]).*?\[N_token=(\d+), P=(\d+), seams: ([^;\]]*?); tiling: ([^\]]*)\]")
NOT_APPLIED_RE = re.compile(r"\[ptx_tp r(\d+)\] NOT applied \((.*?)\); stock behaviour")
BACKEND_RULE = "ptx_tp/dist.py init_from_env: backend = 'nccl' if CUDA is available else 'gloo'"   # the unit's own rule; a rank reads CUDA (its banner names the device)
RANK_LOG_DIR = "tp/torchrun"                            # torchrun's per-rank log root under the records dir (PET_LOG_DIR); rank r's stderr: <root>/<run_id>/attempt_<k>/<r>/stderr.log
PG_TIMEOUT_S = 1800                                     # the ranks' process-group collective timeout (ptx_tp/launch_core.py NCCL_TIMEOUT_S, passed to init_from_env)
BLOCKS: Tuple[int, ...] = (128, 64, 32, 16)             # the unit's layout blocks (ptx_tp/launch.py auto-B: largest with no empty rank; trunk.py: PTX_TP_B)
BLOCKS_SUPPORTED: Tuple[int, ...] = (128, 64, 32)      # the layout grids the line runs P>1 on (tier 2); a run that needs another grid is refused by name
                                                        # (the unit itself would run any grid with a banner; the line never starts an unsupported grid)
ENV_BLOCK = "PTX_TP_B"                                  # the unit's block-size switch (a user value wins over its auto rule)
# The line's block rule: the largest block with NO item of the run in the unit's REPLICATED regime (N < P*B: the unit then runs the stock main
# loop on every rank — not this line's statements, and with the kit's template lever installed that path fails by name) and no empty rank for
# the largest item; an item below P*16 tokens cannot run on the line and is refused by name (the single-GPU line runs it).


def item_tokens(input_path: str) -> List[int]:
    """The launcher's own N estimate (ptx_tp/launch.py _guess_n_tokens: sum of sequence length x count over the entities; ligands 0),
    per item of the Protenix input JSON (the launcher reads the first item only; the line sizes its block for every item of the run)."""
    d = json.load(open(input_path, encoding="utf-8"))
    out = []
    for e in (d if isinstance(d, list) else [d]):
        n = 0
        for ent in e.get("sequences", []):
            for v in ent.values():
                if isinstance(v, dict):
                    n += len(v.get("sequence", "")) * int(v.get("count", 1))
        out.append(n)
    return out


def _layout(n_tokens: int, n: int, block: int):
    if unit_dir() not in sys.path:
        sys.path.insert(0, unit_dir())
    from opt_core.mem.rowpair.dist import Layout                          # the one row-grid arithmetic (the unit's ptx_tp.dist is served by it in every rank)
    return Layout.checked(int(n_tokens), int(n), 0, B=int(block), lever="protenix_v2.tp")   # refuses by name: replicated grid / a rank with zero rows


def block_for(tokens: List[int], n: int) -> Tuple[Optional[int], str]:
    """(B, why) for a run of items with these token counts on N ranks — the unit's own arithmetic (``ptx_tp.dist.Layout``, never
    transcribed): the largest of BLOCKS such that for EVERY item the layout is not replicated AND every rank owns rows (the launcher's
    auto-B loop minus its acceptance of a replicated first item; the unit itself only WARNS on zero-row ranks and names the diffusion /
    confidence tails as not R==0-safe, trunk.py make_layout). (None, the reason) when no block serves every item."""
    if not tokens:
        return None, "no item in the input"
    lo, hi = min(tokens), max(tokens)
    for B in BLOCKS:
        bad = None
        for N in sorted(set(tokens)):
            try:
                _layout(N, n, B)
            except RowpairRefused as e:                                   # replicated grid or a zero-row rank at this B: try the next grid
                bad = f"{N} tokens on {n} ranks at B={B}: {e}"; break
        if bad is None:
            if B not in BLOCKS_SUPPORTED:
                return None, (f"n_gpu={n} at {lo}..{hi} tokens needs the unit's B={B} layout grid (at {lo} tokens the 128/64/32 grids replicate or leave a rank with no rows), "
                              f"which is not supported (P>1 runs on the B=128/64/32 grids): refused — n_gpu={n} requires every item to have at least {n * 32} tokens "
                              f"with no empty rank at B=32; use fewer GPUs for this input")
            return B, f"B={B}: every item ({lo}..{hi} tokens) row-sharded with every rank owning rows on {n} ranks"
    return None, (f"no layout block of {'/'.join(map(str, BLOCKS))} serves every item ({lo}..{hi} tokens) on {n} ranks without the unit's replicated regime "
                  f"(the stock main loop on every rank) or a zero-row rank ({bad}) — run it on the mode's single-GPU line or another N")


def rows_line(tokens: List[int], n: int, block: int) -> str:
    """``rows[<N>]=<r0-r1,...>`` for the smallest and the largest item: the shard every rank owns, from the unit's Layout."""
    out = []
    for N in sorted({min(tokens), max(tokens)}):
        L = _layout(N, n, block); out.append(f"rows[{N}]=" + ",".join(f"{a}-{b}" for a, b in L.bounds))
    return " ".join(out)


def input_path(rest: List[str]) -> Optional[str]:
    """The stock CLI's input file named in the pass-through arguments (-i / --input; the last occurrence wins as click does)."""
    p = None
    for i, t in enumerate(rest):
        if t in ("-i", "--input") and i + 1 < len(rest):
            p = rest[i + 1]
        elif t.startswith("--input="):
            p = t.split("=", 1)[1]
    return p
LAUNCH_LOG = "launch.log"
FINAL_PHASE = "confidence_end"                          # the last phase a rank records per sample (ptx_tp/trunk.py)
RANK_RE = re.compile(r"\[launch_core\] rank (\d+)/(\d+) device=(\S+) name=(.+?) cc=(\S+) mem=(\S+) GiB driver=([^\s\[]+)")
WORLD_RE = re.compile(r"\[launch_core\] P=(\d+) torch=(\S+) cuda=(\S+) nccl=(\S+)")
LAYOUT_RE = re.compile(r"\[launch\] layout block B = (\d+) for N~(\d+), P=(\d+)")
TILING_RE = re.compile(r"\[launch\] TRIMUL/TRIATT env \(PairCore v3 recommended\): (.+)$")


class TpError(Exception):
    """A refusal of the line, by name (the caller maps it to NOT ACTIVE / usage)."""


# ------------------------------------------------------------------------------------------------------------------ selector ----
def split_n_gpu(argv: List[str]) -> Tuple[Optional[int], List[str]]:
    """Take ``--n_gpu P`` / ``--n_gpu=P`` out of ``argv`` -> (P or None when absent, the rest). A value that is not a positive integer is
    refused by name (``opt_core.mem.ngpu.check_n_gpu``)."""
    out: List[str] = []; n: Optional[int] = None; i = 0
    while i < len(argv):
        a = argv[i]
        if a == FLAG_NGPU or a.startswith(FLAG_NGPU + "="):
            if a == FLAG_NGPU:
                if i + 1 >= len(argv):
                    raise TpError(f"{FLAG_NGPU} needs a value")
                raw = argv[i + 1]; i += 2
            else:
                raw = a.split("=", 1)[1]; i += 1
            n = _positive_int(raw, FLAG_NGPU)
            continue
        out.append(a); i += 1
    return n, out


def n_gpu_from_env(environ=None) -> Optional[int]:
    """``PROTENIX_OPT_N_GPU`` -> P (None when unset); a value that is not a positive integer is refused by name."""
    raw = (environ if environ is not None else os.environ).get(ENV_NGPU)
    return None if raw in (None, "") else _positive_int(raw, ENV_NGPU)


def _positive_int(raw, where: str) -> int:
    try:
        return _ngpu.check_n_gpu(raw)
    except ValueError as e:
        raise TpError(f"{where}: {e}")


def selection(flag: Optional[int], environ=None) -> Tuple[int, str]:
    """The GPU count P and where it came from: the CLI flag wins; a flag that disagrees with a set env value is refused (one row of
    record has one selector); neither given = ``DEFAULT_NGPU`` (1, source ``default``) — an absent ``--n_gpu`` and ``--n_gpu 1`` are the
    same selection."""
    e = n_gpu_from_env(environ)
    if flag is not None and e is not None and flag != e:
        raise TpError(f"{FLAG_NGPU} {flag} disagrees with {ENV_NGPU}={e} — drop one of them")
    if flag is not None:
        return _positive_int(flag, FLAG_NGPU), FLAG_NGPU
    return (e, ENV_NGPU) if e is not None else (DEFAULT_NGPU, "default")


def refusal(mode: str, n: int) -> Optional[str]:
    """Why (mode, P) is not a runnable selection, or None: P > 1 belongs to mode ``big`` only — under ``exact`` / ``fast`` / ``off`` it is
    the shared core's sentence ``refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)``; P == 1 runs under any mode
    (the engine's single-GPU path)."""
    try:
        _ngpu.refuse_unless_big(n, mode)
    except _ngpu.NGpuRefused as e:
        return str(e.reason)
    return None


def active_fields(n: int) -> str:
    """``n_gpu=P sharding=rowpair`` for P > 1, ``n_gpu=1 sharding=none`` for the single-GPU line (opt_core.mem.ngpu.active_fields)."""
    return _ngpu.active_fields(n, SCHEME)


def line_selected(mode: str, n: Optional[int]) -> bool:
    """True when the caller asked for this line: mode ``big`` with P > 1."""
    return mode == LINE_MODE and n is not None and int(n) > 1


def unit_dir() -> str:
    """The carried unit's package root (``opt/forward/PTX_TP/PTX_TP_ADDON``): what goes on the ranks' ``PYTHONPATH``."""
    return os.path.join(kits.kit_dir(UNIT), UNIT_PKG_DIR)


def unit_rels() -> List[str]:
    """Every carried file of the unit, as opt-relative paths."""
    return [kits.kit_rel(UNIT, p) for p in kits.tree_files(kits.kit_dir(UNIT))]


AUTOLOAD_PTH = "protenix_opt_autoload.pth"                    # the package's autoload at the wheel root (opt/protenix_opt_autoload.pth): the ranks' activation route


def autoload_installed(executable: Optional[str] = None) -> Optional[str]:
    """The path of the installed autoload ``.pth`` in the site-packages of ``executable`` (this interpreter by default), None when no
    site-packages carries it — then a rank interpreter would import the stock CLI without activating the base mode."""
    import site
    dirs = list(site.getsitepackages()) + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else [])
    if executable and executable != sys.executable:
        out = subprocess.run([executable, "-c", "import site, json; print(json.dumps(site.getsitepackages() + ([site.getusersitepackages()] if site.ENABLE_USER_SITE else [])))"],
                             capture_output=True, text=True)
        dirs = json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else []
    for d in dirs:
        p = os.path.join(d, AUTOLOAD_PTH)
        if os.path.isfile(p):
            return p
    return None


def preflight(n: int) -> dict:
    """What the line needs before a launch: the unit fully present, ``ptx_tp`` importable from it, the package's autoload
    ``.pth`` installed for the rank interpreter (the ranks activate the base mode through it), torch with N visible CUDA devices and the
    NCCL backend. Returns ``{ok, reason, n_gpu, visible_gpus, unit_files, autoload_pth, torch, nccl}``; ``ok`` False names the gap."""
    rep: Dict[str, object] = {"ok": False, "reason": "", "n_gpu": n, "unit": unit_dir()}
    if not os.path.isdir(unit_dir()):
        rep["reason"] = f"unit {UNIT} absent: {unit_dir()}"; return rep
    rels = unit_rels()
    problems = kits.missing_kit_files(rels)
    rep["unit_files"] = len(rels)
    if problems:
        rep["reason"] = f"unit {UNIT}: " + "; ".join(problems[:3]); return rep
    if not os.path.isfile(os.path.join(unit_dir(), "ptx_tp", "launch.py")):
        rep["reason"] = f"unit {UNIT}: ptx_tp/launch.py missing under {unit_dir()}"; return rep
    rep["autoload_pth"] = autoload_installed()
    if not rep["autoload_pth"]:
        rep["reason"] = (f"{AUTOLOAD_PTH} is not installed for {sys.executable}: the ranks activate the base mode through it "
                         f"(pip install -e <tree>/opt; a PYTHONPATH-only package cannot run the line)"); return rep
    try:
        import torch
    except Exception as e:                                      # the pinned stack has torch; a box without it cannot run any mode
        rep["reason"] = f"torch not importable: {e!r}"; return rep
    rep["torch"] = torch.__version__
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    rep["visible_gpus"] = visible
    try:                                                        # fewer visible devices than P: the shared core's sentence `refused: n_gpu=P visible=K` (never shrunk to K)
        _ngpu.refuse_unless_visible(n, visible)
    except _ngpu.NGpuRefused as e:
        rep["reason"] = str(e.reason); rep["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "unset"); return rep
    try:
        rep["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
    except Exception as e:
        rep["reason"] = f"NCCL unavailable: {e!r}"; return rep
    rep["ok"] = True
    return rep


# ------------------------------------------------------------------------------------------------------------------ launch -----
PRE_CUDA_KEYS: Tuple[str, ...] = ("PYTORCH_CUDA_ALLOC_CONF",)   # exports torch reads at CUDA initialisation: a rank initialises CUDA (process group,
                                                               # device binding) BEFORE its entry imports the stock CLI and the autoload activates,
                                                               # so these come from the launching process's dry run of the base mode
LEVER_REPORT = "lever_report.jsonl"                            # <records dir>/tp/: every rank's kit lever report (PTX_LEVER_REPORT), reconciled by the parent
HASHSEED_ENV = _rankdata.HASHSEED_ENV                          # "PYTHONHASHSEED": ONE str-hash seed for the whole launcher -> torchrun -> ranks subtree (``rank_env`` starts
                                                               # from ``rankdata.ranks_env``; torchrun's workers inherit that environment unchanged). The shared
                                                               # core's rule: a decimal integer the caller exported is kept (source ``inherited``); unset, empty,
                                                               # ``random`` or anything else becomes ``0`` (source ``default``; a discarded value is named as
                                                               # ``parent=``). The ranks are separate interpreters: a hash-ordered step of the input featurisation
                                                               # (set / dict iteration over str keys) orders its bytes the same way in every rank only under one
                                                               # seed. Read ONCE per launch (``launch_env``): the ranks' environment, the ACTIVE line's word and the run record's fields


def hashseed_token(n: int, environ=None) -> str:
    """``hashseed=<value> source=<default|inherited> [parent=<repr>] ranks=<P>`` — the shared core's census word (``rankdata.hashseed_word``,
    byte for byte) for the launching process's environment (``environ``; ``os.environ`` when None — the environment ``rank_env`` starts from);
    ``parent=`` only when the caller's value could not be kept, e.g. ``hashseed=0 source=default parent='random' ranks=2``."""
    return _rankdata.hashseed_word(environ, n)


def launch_env(n: int, records: str, exports: Optional[Dict[str, str]] = None, base: Optional[dict] = None,
               det_level: Optional[int] = None) -> Tuple[Dict[str, str], str, Dict[str, str]]:
    """``(env, hashseed_word, hashseed_fields)`` of one launch: the environment every rank starts with (:func:`rank_env` documents it), the
    ACTIVE line's census word ``hashseed=<v> source=default|inherited [parent=<repr>] ranks=P`` and the run record's fields ``{"hashseed",
    "hashseed_source"[, "hashseed_parent"]}`` — all three from ONE reading of the launching process's environment (``base``; ``os.environ``
    when None), by the shared core's rule (``rankdata.ranks_env`` / ``rankdata.hashseed_fields``)."""
    parent = dict(os.environ if base is None else base)                  # the one reading the seed, its word and its record fields come from
    env, word = _rankdata.ranks_env(parent, n)                           # the shared core's base environment of the ranks: the caller's environment with the launch's one
    fields = _rankdata.hashseed_fields(parent)                           # str-hash seed exported (a decimal integer the caller exported is kept, anything else is 0)
    env["PROTENIX_OPT"] = BASE_MODE
    env.pop(ENV_NGPU, None)                                     # the ranks are not a selection: PROTENIX_OPT_N_GPU in a rank would re-enter this line
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = unit_dir() + (":" + pp if pp else "")
    env["PTX_TP_PHASE_LOG"] = os.path.join(records, PHASE_LOG_DIR)
    env["PET_LOG_DIR"] = os.path.join(records, RANK_LOG_DIR); env["PET_REDIRECTS"] = "3"; env["PET_TEE"] = "3"   # torchrun (its PET_* env form of --log-dir/--redirects/--tee): every
                                                                                                                # rank's stderr in its own file <out>/tp/torchrun/<run>/attempt_0/<rank>/stderr.log — the ledger source
                                                                                                                # the shared pipe (launch.log) keeps a teed copy for the eye
    env["PTX_LEVER_REPORT"] = os.path.join(records, PHASE_LOG_DIR, LEVER_REPORT)
    env["PYTHONUNBUFFERED"] = "1"
    for k in PRE_CUDA_KEYS:
        if exports and exports.get(k):
            env[k] = exports[k]
    env.update(TP_PRE)                                          # the graph levers stay uninstalled in every rank (TP_DROPPED)
    env[tp_route.ENV] = tp_route.WORD                          # every rank routes the unit's dist/blockreduce/bcast/contract names to opt_core.mem.rowpair
    env.update(BIND_ENV)                                         # the bound diffusion seam's rows-only pair-bias cache budget (core word; the unit's PTX_TP_DIFFCACHE_GB=8 envelope)
    for k, v in RANK_ENV_DEFAULTS.items():                       # the ranks' defaults a caller's own value overrides (the per-rank CPU-thread cap)
        if not (env.get(k) or "").strip():
            env[k] = v
    env[NOISE_SYNC_ENV] = noise_sync(det_level if det_level is not None else int(env.get("PTX_DET", "0") == "1"))   # det 0 -> bcast, det 1 -> guard (NOISE_SYNC_BY_DET)
    why = gate_refusal(env)                                     # a caller's open gate is refused, never overridden silently (cli.tp_pred_route refuses before the launch)
    if why:
        raise TpError(why)
    env.update(TP_GATES)                                        # every size gate of the unit forced to its sharded value (never setdefault: the line's statement)
    return env, word, fields


def rank_env(n: int, records: str, exports: Optional[Dict[str, str]] = None, base: Optional[dict] = None, det_level: Optional[int] = None) -> Dict[str, str]:
    """The environment every rank starts with: the base mode for the autoload (``PROTENIX_OPT=fast``), the unit on ``PYTHONPATH``
    (first), the phase-log and lever-report paths, one str-hash seed for every rank interpreter (``PYTHONHASHSEED``, the shared core's
    ``rankdata.ranks_env``: a decimal integer the caller exported is kept, anything else is ``0``), and the base mode's pre-CUDA exports
    (``PRE_CUDA_KEYS`` out of ``exports``, the dry run's resolved environment). Everything else the launcher sets itself (``PTX_TP=N`` and
    its PairCore defaults) or the rank's own activation exports. The environment of :func:`launch_env`."""
    return launch_env(n, records, exports=exports, base=base, det_level=det_level)[0]


def gate_refusal(environ=None) -> Optional[str]:
    """Why the caller's environment cannot run the line, or None: a size gate of the unit (``TP_GATES``) set to another value than the
    line's forced one — ``refused: gate <NAME>=<value> under big --n_gpu P: every Mode-S path is sharded at every N (<NAME>=<forced>); a
    size-gated tensor-parallel mode is not a setting of this line``. An unset gate or one already at the forced value passes."""
    env = os.environ if environ is None else environ
    for k, v in TP_GATES.items():
        have = (env.get(k) or "").strip()
        if have and have != v:
            return (f"refused: gate {k}={have} under {LINE_MODE} {FLAG_NGPU} P: every Mode-S path is sharded at every N ({k}={v}); "
                    f"a size-gated tensor-parallel mode is not a setting of this line")
    return None


def p1_levers_token() -> str:
    """``p1_levers=<lever>:<per_rank|replaced_by_rowpair:<statement>|unit_release_points>,...`` — big's single-GPU memory levers under the line."""
    return "p1_levers=" + ",".join(f"{k}:{v}" for k, v in P1_LEVERS_UNDER_TP.items())


def gates_token() -> str:
    """``gates=<NAME>=<value>,...`` — the forced gates as the ACTIVE line states them (``TP_GATES``, in order)."""
    return "gates=" + ",".join(f"{k}={v}" for k, v in TP_GATES.items())


def census_token() -> str:
    """``sharded_rows=<n> replicated_by_design=<names>`` — the schedule census in the ACTIVE line's words (``CENSUS``; the manifest carries the lists)."""
    rep = ";".join(x.split(" [")[0].split(" (")[0] for x in CENSUS["replicated_by_design"])
    return f"sharded_rows={len(CENSUS['sharded_rows'])}_pair_tensors replicated_by_design={rep.replace(' ', '_')}"


def command(n: int, rest: List[str], records: str, label: str = LINE) -> List[str]:
    """``python -m ptx_tp.launch --nproc N --lazy-relp --lift-guard --label <label> --out <records dir>/tp -- pred <stock arguments>``."""
    return [sys.executable, "-m", LAUNCHER_MODULE, "--nproc", str(n), *LINE_ARGS, "--label", label, "--out", os.path.join(records, PHASE_LOG_DIR),
            "--", "pred", *rest]


def run(n: int, rest: List[str], records: str, label: str = LINE, env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Launch the line; the launcher's stdout+stderr stream to the console and to ``<records dir>/tp/launch.log``. Returns (rc, log path)."""
    tp_dir = os.path.join(records, PHASE_LOG_DIR); os.makedirs(tp_dir, exist_ok=True)
    log_path = os.path.join(tp_dir, LAUNCH_LOG)
    cmd = command(n, rest, records, label)
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n"); log.flush()
        proc = subprocess.Popen(cmd, env=env if env is not None else rank_env(n, records), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stderr.write(line); sys.stderr.flush(); log.write(line)
        rc = proc.wait()
    return rc, log_path


# ------------------------------------------------------------------------------------------------------------------ events -----
def _seams(text: str) -> Dict[str, str]:
    """The seams found in a ledger line (``SEAM_RE``): the ranks share one stderr pipe under torchrun, so another writer's line can be
    glued to a ledger line — the value is only ever ``tp``; a seam absent from the line (or carrying another word) is a named parse
    failure (``events`` records it), never read as ``tp``."""
    return {k: v for k, v in SEAM_RE.findall(text)}


def rank_log_files(records: str) -> Dict[int, List[str]]:
    """{rank: [stdout.log, stderr.log]} of torchrun's per-rank log files under <records dir>/tp/torchrun (PET_LOG_DIR): the newest attempt of
    the one run; the unit's ledger lines are on both streams (the effective-seams line is a stdout line, the APPLIED line a stderr line).
    {} when torchrun wrote none; a rank with only one stream counts as present (its lines are what its files hold)."""
    out: Dict[int, List[str]] = {}; newest: Dict[int, float] = {}
    for p in glob.glob(os.path.join(records, RANK_LOG_DIR, "*", "attempt_*", "*", "std*.log")):
        try:
            r = int(os.path.basename(os.path.dirname(p)))
        except ValueError:
            continue
        attempt_dir = os.path.dirname(os.path.dirname(p)); t = os.path.getmtime(attempt_dir)
        if r not in out or t > newest[r]:
            out[r] = [p]; newest[r] = t
        elif t == newest[r]:
            out[r].append(p)
    return {r: sorted(ps) for r, ps in out.items()}


def events(records: str, n: int, block: Optional[int] = None) -> dict:
    """The named events of a run, read from what the ranks wrote: world size and versions (``[launch_core] P=...``), every rank's
    banner (device, name, cc, memory), the layout block and the tiling line (``[launch] ...``), every rank's APPLIED ledger line (the
    unit's seams as imported: every seam ``tp``; a seam not readable as ``tp`` is a parse failure by name), the seams record of the forward
    (``per-rank peak alloc ... seams: ...``), the unit's ``NOT applied`` line (a refusal by name), the collective backend by the unit's own rule and its process-group watchdog, the row-shard map from the unit's
    ``Layout``, the run's ``runmeta`` record (per-rank peak allocation, forward/trunk/diffusion/confidence seconds), and per rank the
    phase log (``phases_rank<r>.jsonl``: last phase, samples finished = ``confidence_end`` records, peak allocated GiB). ``ok`` is True
    only when the world size equals N, every rank 0..N-1 reported its banner and its APPLIED line with every seam ``tp``, no rank
    printed NOT applied, and every rank's log ends at the final phase."""
    tp_dir = os.path.join(records, PHASE_LOG_DIR)
    ev: Dict[str, object] = {"line": LINE, "n_gpu": n, "sharding": SCHEME, "base_mode": BASE_MODE, "line_args": list(LINE_ARGS), "stock_arg_delta": list(STOCK_ARG_DELTA),
                             "strategy": STRATEGY, "impl": IMPL, "dropped_levers": list(TP_DROPPED), "world_size": None, "ranks": {}, "backend": None,
                             "pg_timeout_s": PG_TIMEOUT_S, "layout": None, "tiling": None,
                             "shard_map": None, "seams_applied": {}, "seams_effective": None, "parse_failures": {},
                             "regime": None, "replicated": [], "empty_ranks": [],
                             "not_applied": {}, "runmeta": None, "phases": {}, "bound": {}, "bound_expected": list(tp_route.BIND), "routed": {}, "routed_expected": list(tp_route.ROUTES),
                             "launcher": LAUNCHER, "ok": False, "reason": ""}
    log_path = os.path.join(tp_dir, LAUNCH_LOG)
    if os.path.isfile(log_path):
        text = open(log_path, encoding="utf-8", errors="replace").read()   # the ranks write stderr concurrently: two banners can share a line
        pipe = text
        rank_files = rank_log_files(records)                                  # the ledger source: torchrun's per-rank log files (no interleaving, no loss)
        if rank_files and set(rank_files) == set(range(n)):
            text = "\n".join(open(p, encoding="utf-8", errors="replace").read() for _, ps in sorted(rank_files.items()) for p in ps)
            ev["ledger_source"] = {"form": "per_rank_files", "files": {str(r): ps for r, ps in sorted(rank_files.items())}, "shared_pipe": log_path}
        else:                                                                 # the shared pipe only (a lost or glued line reads as missing / a named parse failure)
            ev["ledger_source"] = {"form": "shared_pipe", "files": {str(r): ps for r, ps in sorted(rank_files.items())}, "shared_pipe": log_path,
                                   "note": f"per-rank log files present for ranks {sorted(rank_files)} of {n}: the ranks' ledger read from the shared pipe"}
        for m in WORLD_RE.finditer(text):
            ev["world_size"] = int(m.group(1)); ev["backend"] = {"collective": "nccl", "rule": BACKEND_RULE, "nccl": m.group(4), "torch": m.group(2), "cuda": m.group(3)}
        for m in RANK_RE.finditer(text):
            ev["ranks"][int(m.group(1))] = {"world": int(m.group(2)), "device": m.group(3), "name": m.group(4), "cc": m.group(5), "mem_gib": m.group(6)}
        for m in LAYOUT_RE.finditer(pipe):                                    # the launcher's own lines (the parent process): the pipe
            ev["layout"] = {"block": int(m.group(1)), "n_tokens_estimate": int(m.group(2)), "world": int(m.group(3))}
        for m in TILING_RE.finditer(pipe):
            ev["tiling"] = m.group(1).strip()
        for m in REPLICATED_RE.finditer(text):                                   # the unit decided on the REAL N: the stock loop ran on every rank
            ev["replicated"].append({"n_tokens": int(m.group(1)), "p_times_b": int(m.group(2))})
        for m in ZERO_ROWS_RE.finditer(text):
            ev["empty_ranks"].append({"block": int(m.group(1)), "ranks": json.loads(m.group(2)), "n_tokens": int(m.group(3)), "world": int(m.group(4))})
        for m in APPLIED_RE.finditer(text):
            r = int(m.group(1)); seams = _seams(m.group(7))
            ev["seams_applied"][r] = {"P": int(m.group(2)), "feat": m.group(3), "relp": m.group(4), "zinit_recompute": m.group(5), "extras": m.group(6), "seams": seams}
            for k in SEAMS:
                if k not in seams:
                    ev["parse_failures"][f"rank{r}:{k}"] = f"seam {k}=tp not readable on rank {r}'s APPLIED line (shared stderr pipe: {m.group(7)[:80]!r})"
        for m in EFFECTIVE_RE.finditer(text):
            ev["seams_effective"] = _seams(m.group(4))
            ev["runmeta"] = dict(ev["runmeta"] or {}, per_rank_peak_alloc_gib_line=m.group(1), n_token=int(m.group(2)))
        for m in NOT_APPLIED_RE.finditer(text):
            ev["not_applied"][int(m.group(1))] = m.group(2)
        if rank_files and set(rank_files) == set(range(n)):                   # the route lines per rank (each rank's own log files)
            for r, ps in sorted(rank_files.items()):
                ev["routed"][r] = sorted(set(sum((tp_route.route_lines(open(p, encoding="utf-8", errors="replace").read()) for p in ps), [])))
                ev["bound"][r] = sorted(set(sum((tp_route.bind_lines(open(p, encoding="utf-8", errors="replace").read()) for p in ps), [])))
        else:
            ev["routed"] = {"pipe": sorted(set(tp_route.route_lines(text)))}
            ev["bound"] = {"pipe": sorted(set(tp_route.bind_lines(text)))}
    for path in sorted(glob.glob(os.path.join(tp_dir, "runmeta_*.json"))):
        try:
            rm = json.load(open(path, encoding="utf-8"))
            ev["runmeta"] = dict(ev["runmeta"] or {}, **{k: rm.get(k) for k in ("per_rank_peak_alloc_GiB", "single_card_peak_GiB", "tiling", "wall_model_forward_s", "t_trunk_s", "t_diffusion_s", "t_confidence_s") if k in rm})
        except (OSError, ValueError) as e:
            ev["runmeta"] = dict(ev["runmeta"] or {}, error=repr(e))
    n_eff = (ev["runmeta"] or {}).get("n_token")                           # the unit's EFFECTIVE N_token (its own line), never the estimate
    b_eff = block or (ev["layout"] or {}).get("block")                     # the block the route exported (PTX_TP_B), else the launcher's auto-B line
    if n_eff and b_eff:
        try:
            ev["shard_map"] = shard_map(n_eff, n, b_eff)
        except Exception as e:  # noqa: BLE001 — the unit's Layout is the rule; a failure to read it is named, never a silent gap
            ev["shard_map"] = {"error": repr(e)}
    for path in sorted(glob.glob(os.path.join(tp_dir, "phases_rank*.jsonl"))):
        r = int(re.search(r"phases_rank(\d+)", path).group(1))
        last = None; done = 0; peak = 0.0; count = 0
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            count += 1; last = rec.get("phase")
            done += 1 if last == FINAL_PHASE else 0
            peak = max(peak, float((rec.get("cuda") or {}).get("max_alloc_GiB") or 0.0))
        ev["phases"][r] = {"records": count, "last": last, "samples_finished": done, "max_alloc_GiB": round(peak, 3)}
    missing_banner = [r for r in range(n) if r not in ev["ranks"]]
    missing_applied = [r for r in range(n) if r not in ev["seams_applied"]]
    missing_log = [r for r in range(n) if r not in ev["phases"]]
    unfinished = [r for r, p in ev["phases"].items() if p["last"] != FINAL_PHASE]
    if ev["world_size"] != n:                                                     # the launcher's rank count as the model processes report it, never the request
        ev["reason"] = f"reason=n_gpu_mismatch requested={n} active={ev['world_size'] if ev['world_size'] is not None else 'unread'} (the ranks' banners)"
    elif missing_banner:
        ev["reason"] = f"ranks without a banner: {missing_banner}"
    elif ev["not_applied"]:
        ev["reason"] = "the unit refused to apply in rank(s) " + ", ".join(f"{r}: {why}" for r, why in sorted(ev["not_applied"].items()))
    elif missing_applied:
        ev["reason"] = f"ranks without the unit's APPLIED line: {missing_applied}"
    elif ev["shard_map"] is None:
        ev["reason"] = "the unit's effective N_token line is absent: no shard map recorded"
    elif "pipe" not in ev["routed"] and any(set(v) != set(tp_route.ROUTES) for v in ev["routed"].values()):
        ev["reason"] = "ranks whose carried-unit layers did not route to the core: " + "; ".join(
            f"rank{r}: routed {v or 'none'} of {list(tp_route.ROUTES)}" for r, v in sorted(ev["routed"].items()) if set(v) != set(tp_route.ROUTES))
    elif "pipe" in ev["routed"] and set(ev["routed"]["pipe"]) != set(tp_route.ROUTES):
        ev["reason"] = f"the carried unit's layers did not route to the core (shared pipe read): routed {ev['routed']['pipe'] or 'none'} of {list(tp_route.ROUTES)}"
    elif "pipe" not in ev["bound"] and any(set(v) != set(tp_route.BIND) for v in ev["bound"].values()):
        ev["reason"] = "ranks whose engine seams were not bound to the kit's bindings on the core: " + "; ".join(
            f"rank{r}: bound {v or 'none'} of {list(tp_route.BIND)}" for r, v in sorted(ev["bound"].items()) if set(v) != set(tp_route.BIND))
    elif "pipe" in ev["bound"] and set(ev["bound"]["pipe"]) != set(tp_route.BIND):
        ev["reason"] = f"the engine seams were not bound to the kit's bindings (shared pipe read): bound {ev['bound']['pipe'] or 'none'} of {list(tp_route.BIND)}"
    elif ev["parse_failures"]:
        ev["reason"] = "ledger lines not readable (a parse failure, named — not a fallback): " + "; ".join(f"{k} {v}" for k, v in sorted(ev["parse_failures"].items()))
    elif ev["replicated"]:
        ev["regime"] = "replicated"
        ev["reason"] = "the unit ran its REPLICATED regime (the stock main loop on every rank, not the line's statements) for " + ", ".join(f"N={x['n_tokens']} < P*B={x['p_times_b']}" for x in ev["replicated"])
    elif ev["empty_ranks"]:
        ev["reason"] = "zero-row ranks (the unit names its diffusion/confidence tails as not R==0-safe): " + ", ".join(f"B={x['block']} N={x['n_tokens']} P={x['world']} ranks {x['ranks']}" for x in ev["empty_ranks"])
    elif missing_log:
        ev["reason"] = f"ranks without a phase log: {missing_log}"
    elif unfinished:
        ev["reason"] = f"ranks that did not reach {FINAL_PHASE}: {sorted(unfinished)}"
    else:
        ev["ok"] = True
    ev["ranks"] = {str(k): v for k, v in ev["ranks"].items()}; ev["phases"] = {str(k): v for k, v in ev["phases"].items()}
    if ev["regime"] is None:
        ev["regime"] = "tp" if ev["ok"] else None
    ev["seams_applied"] = {str(k): v for k, v in ev["seams_applied"].items()}; ev["not_applied"] = {str(k): v for k, v in ev["not_applied"].items()}
    return ev


def bound_count(ev: dict) -> int:
    """Ranks whose TP-BIND lines name every module of ``tp_route.BIND`` (the shared-pipe read counts as one when complete)."""
    r = ev.get("bound") or {}
    if "pipe" in r:
        return 1 if set(r["pipe"]) == set(tp_route.BIND) else 0
    return sum(1 for v in r.values() if set(v) == set(tp_route.BIND))


def routed_count(ev: dict) -> int:
    """Ranks whose TP-ROUTE lines name every module of ``tp_route.ROUTES`` (the shared-pipe read counts as one when complete)."""
    r = ev.get("routed") or {}
    if "pipe" in r:
        return 1 if set(r["pipe"]) == set(tp_route.ROUTES) else 0
    return sum(1 for v in r.values() if set(v) == set(tp_route.ROUTES))


def final_line(prefix: str, rep: dict, ev: dict, n: int) -> str:
    """The ONE ``FINAL`` line of a tp run: the mode's reconciled grammar (``report.final_line``) with ``line=tp n_gpu=P sharding=rowpair``
    after the mode and the run's events appended — never the single line's grammar beside it (a regex for the single line must not match a tp run)."""
    from . import report as _report
    base = _report.final_line(dict(rep, n_gpu=n, sharding=SCHEME))
    head = f"{prefix} FINAL mode={LINE_MODE} "
    assert base.startswith(head), base
    return (f"{head}line={LINE} " + base[len(head):] + f" world_size={ev['world_size']} ranks={len(ev['ranks'])} "
            f"finished={sum(1 for p in ev['phases'].values() if p['last'] == FINAL_PHASE)} backend={(ev['backend'] or {}).get('collective')} "
            f"seams={','.join(f'{k}={v}' for k, v in (ev['seams_effective'] or {}).items()) or 'unread'} "
            f"routed={routed_count(ev)}/{n} bound={bound_count(ev)}/{n} "
            f"diffusion_regime={(rep.get('diffusion_regime') or {}).get('mode', 'unread')},{(rep.get('diffusion_regime') or {}).get('precision', 'unread')} "
            f"ledger={(ev.get('ledger_source') or {}).get('form', 'unread')} ok={str(ev['ok']).lower()}"
            + (f" reason={ev['reason']}" if ev["reason"] else ""))


def rank_records(path: str) -> Dict[int, dict]:
    """Every rank's lines of the shared lever report merged per pid (the kit's own records — clisampler, the trunk record with the
    counters, stackgraph, fpf_trimul_v4 ... — and this package's ``protenix_opt`` activation record appended at exit;
    ``report.read_lever_report`` is the one reader), keyed by pid. A pid without a ``protenix_opt`` record is a rank that died before
    its exit tally."""
    from . import report as _report
    out: Dict[int, dict] = {}
    if not os.path.isfile(path):
        return out
    pids: List[int] = []
    for raw in open(path, encoding="utf-8", errors="replace"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("pid") is not None and int(rec["pid"]) not in pids:
            pids.append(int(rec["pid"]))
    for pid in pids:
        merged = _report.read_lever_report(path, pid)
        if merged:
            out[pid] = merged
    return out


def _count(rec: dict, key: str, paths: Tuple[Tuple[str, ...], ...]) -> Optional[int]:
    """The execution count of one tally (the record's key, the counter paths summed; a bare number when no path); None when absent."""
    v = rec.get(key)
    if v is None:
        return None
    if not paths:
        return int(v) if isinstance(v, (int, float)) else None
    total = 0; seen = False
    for p in paths:
        x = v
        for k in p:
            x = x.get(k) if isinstance(x, dict) else None
        if isinstance(x, (int, float)):
            total += int(x); seen = True
    return total if seen else None


def execution(records: Dict[int, dict], levers: List[str]) -> dict:
    """Per base-mode lever, what the ranks' tallies say it did: ``executed`` {lever: {pid: count}} (count > 0 in every rank),
    ``inert_under_tp`` {lever: {pid: count}} (a zero count in some rank: installed, never reached — the unit's own kernels ran that
    statement), ``env_only`` [levers without a counter: environment switches the kit reads, no execution tally exists],
    ``no_record`` [levers whose tally record no rank wrote]."""
    out = {"executed": {}, "inert_under_tp": {}, "env_only": [], "no_record": []}
    for name in levers:
        if name not in TALLY:
            out["env_only"].append(name); continue
        key, paths = TALLY[name]
        counts = {str(pid): _count(rec, key, paths) for pid, rec in sorted(records.items())}
        if all(c is None for c in counts.values()):
            out["no_record"].append(name); continue
        if all(c for c in counts.values() if c is not None) and any(c for c in counts.values()):
            out["executed"][name] = counts
        else:
            out["inert_under_tp"][name] = counts
    return out


def reconcile_ranks(rep: dict, path: str, n: int) -> dict:
    """The line's report brought up to date with every rank's records: ``levers_applied`` = the base-mode levers the ranks report
    applied in EVERY rank minus the line's dropped levers (``levers_dropped_by_line``: TP_DROPPED, left uninstalled by TP_PRE — never a
    fallback), ``levers_fallback`` = every lever any rank fell back on (with the rank's reason), ``partial`` follows; a rank without a
    record is a fallback by name (``rank_record_missing``); ``execution`` = what the ranks' tallies say each applied lever did
    (``levers_executed`` / ``levers_inert_under_tp`` / ``levers_env_only``). ``reconciled`` names the records seen."""
    recs = rank_records(path)
    r = dict(rep)
    r["levers_dropped_by_line"] = [x for x in (rep.get("levers_applied") or []) if x in TP_DROPPED]
    r["dropped_reasons"] = {x: f"not part of the {LINE} line: {TP_PRE} on every rank" for x in r["levers_dropped_by_line"]}
    if not recs:
        r.update(levers_fallback=list(r.get("levers_fallback") or []) + ["rank_record_missing"], partial=True,
                 fallback_reasons={**(r.get("fallback_reasons") or {}), "rank_record_missing": f"no rank wrote its activation record to {path}"},
                 execution=None, reconciled={"records": [], "ranks_reported": 0, "ranks": n, "moves": {}})
        return r
    applied = None; fallback: Dict[str, str] = {}
    for pid, rec in sorted(recs.items()):
        po = rec.get("protenix_opt")
        if not isinstance(po, dict):
            fallback["rank_record_missing"] = f"rank pid {pid}: no activation record (the rank died before its exit tally)"; continue
        on = set(po.get("levers_applied") or [])
        applied = on if applied is None else applied & on
        for name in po.get("levers_fallback") or []:
            if name in TP_DROPPED:
                continue                                        # dropped by the line, by name: never a fallback
            fallback[name] = f"rank pid {pid}: " + str((po.get("fallback_reasons") or {}).get(name) or "no reason recorded")
        if not po.get("active"):
            fallback["rank_not_active"] = f"rank pid {pid}: {po.get('reason') or 'not active'}"
    if len(recs) < n:
        fallback["rank_record_missing"] = f"{len(recs)} of {n} ranks wrote a record to {path}"
    for name in (set(r.get("levers_applied") or []) - set(TP_DROPPED) - (applied or set())):
        fallback.setdefault(name, "not applied in every rank")
    r["levers_applied"] = sorted((applied or set()) - set(TP_DROPPED))
    r["levers_fallback"] = sorted(fallback)
    r["fallback_reasons"] = {**(r.get("fallback_reasons") or {}), **fallback}
    r["partial"] = bool(fallback)
    ex = execution(recs, r["levers_applied"])
    r["execution"] = ex
    r["levers_executed"] = sorted(ex["executed"]); r["levers_inert_under_tp"] = sorted(ex["inert_under_tp"]); r["levers_env_only"] = sorted(ex["env_only"] + ex["no_record"])
    r["diffusion_regime"] = diffusion_regime(recs)
    r["gates"] = dict(TP_GATES); r["p1_levers"] = dict(P1_LEVERS_UNDER_TP); r["census"] = {k: list(v) for k, v in CENSUS.items()}
    if r["diffusion_regime"]["mode"] in REGIME_FALLBACK_MODES:                # the unit's regime statement is part of the run's record: absent or split = a named fallback
                                                                               # (never silence); replicated = the diffusion gate open in a rank although the line forces it (TP_GATES)
        fallback[f"diffusion_regime_{r['diffusion_regime']['mode']}"] = (f"the unit's diffusion regime is {r['diffusion_regime']['mode']} across the ranks' records "
                                                                        f"(the line forces PTX_TP_DIFF_REPLICATE_BELOW={TP_GATES['PTX_TP_DIFF_REPLICATE_BELOW']}: tp at every N): "
                                                                        + json.dumps(r["diffusion_regime"]["ranks"], sort_keys=True)[:200])
        r["levers_fallback"] = sorted(set(r["levers_fallback"]) | {f"diffusion_regime_{r['diffusion_regime']['mode']}"}); r["fallback_reasons"] = {**r["fallback_reasons"], **fallback}; r["partial"] = True
    if r["diffusion_regime"].get("precision") == "bf16":                       # the sampler under bf16 autocast in the ranks although the guard lift keeps it fp32 at every N: a named fallback
        fallback["diffusion_regime_bf16"] = ("the unit's diffusion sampler ran under bf16 autocast across the ranks' records (the line's guard lift keeps "
                                             "skip_amp.sample_diffusion True at every N: fp32): " + json.dumps(r["diffusion_regime"]["ranks"], sort_keys=True)[:200])
        r["levers_fallback"] = sorted(set(r["levers_fallback"]) | {"diffusion_regime_bf16"}); r["fallback_reasons"] = {**r["fallback_reasons"], **fallback}; r["partial"] = True
    r["reconciled"] = {"records": [rank_label(pid, rec) for pid, rec in sorted(recs.items())], "ranks_reported": len(recs), "ranks": n,
                       "moves": {**{name: "applied -> fallback" for name in fallback if name in (rep.get("levers_applied") or [])},
                                 **{name: "applied -> dropped_by_line" for name in r["levers_dropped_by_line"]}}}
    return r


def rank_label(pid: int, rec: dict) -> str:
    """``rank<r>:pid<pid>`` from the unit's own rank id in the record (``ptx_tp.rank``), ``rank?:pid<pid>`` when the record carries none."""
    r = (rec.get("ptx_tp") or {}).get("rank") if isinstance(rec.get("ptx_tp"), dict) else None
    return f"rank{r if r not in (None, '') else '?'}:pid{pid}"


REGIME_FALLBACK_MODES: Tuple[str, ...] = ("unread", "disagree", "replicated")   # diffusion regimes that make a run partial by name: no rank stated it, the ranks
                                                                                # disagree, or the unit ran its replicated diffusion statement (the line forces tp)


def diffusion_regime(recs: Dict[int, dict]) -> dict:
    """The diffusion regime the unit ran, from every rank's own record (``report.unit_record``: ``diffusion.report()["modes"]
    ["sample_diffusion"]`` — replicated|tp, fp32|bf16 (skip_amp), N, P): the one regime when every rank states the same, else
    ``mode='disagree'`` naming each rank; ``mode='unread'`` when no rank's record carries it. The line forces ``tp`` (``TP_GATES``) and
    ``fp32`` (the guard lift keeps ``skip_amp.sample_diffusion`` True at every N); the record states what each rank ran, and
    ``reconcile_ranks`` names any other regime a fallback (``REGIME_FALLBACK_MODES``, ``diffusion_regime_bf16``)."""
    per: Dict[str, dict] = {}
    for pid, rec in sorted(recs.items()):
        u = rec.get("ptx_tp") if isinstance(rec.get("ptx_tp"), dict) else None
        d = (u or {}).get("diffusion")
        if isinstance(d, dict):
            per[rank_label(pid, rec)] = {k: d.get(k) for k in ("mode", "precision", "N", "P")}
    if not per:
        return {"mode": "unread", "precision": "unread", "N": None, "P": None, "ranks": {}}
    stmts = {json.dumps(v, sort_keys=True) for v in per.values()}
    if len(stmts) == 1:
        one = next(iter(per.values())); return {**one, "ranks": per}
    return {"mode": "disagree", "precision": "disagree", "N": None, "P": None, "ranks": per}


def execution_line(prefix: str, rep: dict) -> str:
    """``[protenix-opt] EXECUTION line=tp levers_executed=<name:count,...> levers_inert_under_tp=<name:count,...> env_only=<names>
    dropped_by_line=<names>`` — what the ranks' tallies say the applied levers did (counts = rank 0's; every rank's in the manifest)."""
    ex = rep.get("execution") or {}
    def one(k, v):
        return f"{k}:{list(v.values())[0] if v else 'n/a'}"
    def fmt(d):
        return ",".join(one(k, v) for k, v in sorted(d.items())) or "none"
    return (f"{prefix} EXECUTION line={LINE} levers_executed={fmt(ex.get('executed') or {})} levers_inert_under_tp={fmt(ex.get('inert_under_tp') or {})} "
            f"env_only={','.join(rep.get('levers_env_only') or []) or 'none'} dropped_by_line={','.join(rep.get('levers_dropped_by_line') or []) or 'none'}")


def shard_map(n_tokens: int, n: int, block: int = 128) -> dict:
    """The row shard every rank owns for N tokens under the unit's own layout rule (``ptx_tp.dist.Layout``: nb = ceil(N/B) blocks,
    bpr = ceil(nb/P) blocks per rank; replicated when N < P*B), for the manifest — the unit's rule, never transcribed."""
    if unit_dir() not in sys.path:
        sys.path.insert(0, unit_dir())
    from opt_core.mem.rowpair.dist import Layout
    try:
        L = Layout.checked(int(n_tokens), int(n), 0, B=int(block), lever="protenix_v2.tp")
    except RowpairRefused as e:                                           # the grid cannot shard this input on n ranks (block_for refuses such a run before launch)
        return {"n_tokens": int(n_tokens), "world": int(n), "block": int(block), "replicated": True, "rows_per_rank": [], "refused": str(e)}
    return {"n_tokens": L.N, "world": L.P, "block": L.B, "replicated": bool(L.replicated), "rows_per_rank": [[int(a), int(b)] for a, b in L.bounds]}


def active_line(prefix: str, n: int, base_rep: dict, det_level: Optional[int] = None, hashseed: Optional[str] = None) -> str:
    """``[protenix-opt] ACTIVE mode=big line=tp n_gpu=P sharding=rowpair impl=ptx_tp ... launcher=torchrun_loopback hashseed=<v>
    source=<default|inherited> [parent=<repr>] ranks=P base=fast levers_installed=<the base mode's levers the ranks install> dropped_by_line=<TP_DROPPED> ...``
    — the line's activation status, printed by the parent once the preflight passed and the base mode's dry run is active, before any rank
    starts; an installation statement — what executed is the EXECUTION line after the run. The ``n_gpu=P sharding=rowpair`` text is the
    shared core's (``opt_core.mem.ngpu``); ``hashseed= source= [parent=] ranks=`` is the one str-hash seed of the P rank interpreters — the
    word of the launch (``hashseed``: :func:`launch_env`'s, read once with the ranks' environment; ``hashseed_token`` when None)."""
    installed = [x for x in (base_rep.get("levers_applied") or []) if x not in TP_DROPPED]
    return (f"{prefix} ACTIVE mode={LINE_MODE} line={LINE} {active_fields(n)} impl={IMPL} routed={','.join(tp_route.ROUTED)}->{tp_route.CORE_PKG} bound={','.join(tp_route.BOUND)}->{tp_route.BIND_PKG} across_P={ACROSS_P} inplace_chunk={INPLACE_CHUNK} noise_sync={noise_sync(det_level)} launcher={LAUNCHER} "
            f"{hashseed if hashseed is not None else hashseed_token(n)} base={BASE_MODE} protenix={base_rep.get('protenix_version')} "
            f"levers_installed={','.join(installed) or 'none'} dropped_by_line={','.join(TP_DROPPED)} line_args={','.join(LINE_ARGS)} strategy={STRATEGY} "
            f"{gates_token()} {p1_levers_token()} {census_token()} "
            f"(what executed: the EXECUTION line after the run)")
