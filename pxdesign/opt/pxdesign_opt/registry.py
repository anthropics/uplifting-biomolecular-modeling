"""The levers: what each one changes, its class, its numerics tier, the switch that governs it and where it lives.

One lever module, `opt/forward/hoist/pxd_xattempt/hoist.py`, computes five families of step-invariant tensors ONCE per
`sample_diffusion` call — by calling the stock sub-modules on the stock inputs — and replays them at every denoiser step through
class-level patches with stock fall-through (`install()`, l.413-446). Every family is tier 1 (byte-identical to stock under the
deterministic recipe). The kit's switches are read by the
kit's own code (`hoist.py` l.401, 415, 433; `KIT_SWITCHES`); `modes.KIT_MODES` says which values a mode exports.

The package-lever hook (`stack.py`, after `load_checkpoint()`) applies the levers of the package's own (`PACKAGE_LEVERS`; `sizeceil.py`,
`rowpipe.py`, `precision.py`, `sdedup.py`): the three size levers (featdiet and padmask on `fast` and `big`, rowpipe on `big`) and the two
tolerance-class levers both `fast` and `big` select — `tf32` (the numerics policy) and `sdedup` (the single-conditioning row dedup).
Nothing here is applied by this file: `stack.py` applies.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

KIT_RELPATH = "opt/forward/hoist"                       # the lever kit: carried in opt/forward/hoist
CARRIED_KITS = (KIT_RELPATH,)                           # each carried directory (presence-checked at activation, list_kit_files below)
LEVER_MODULE = "pxd_xattempt.hoist"
LEVER_FILE = "pxd_xattempt/hoist.py"
KIT_VERSION_LINE = "[pxd_hoist v1.2.2]"                 # the kit's own print at each prepare (hoist.py l.177): the activation evidence beside ACTIVE
KIT_MODULE_PREFIXES = ("pxd_xattempt",)                  # the carried kit's importable package: a stock process holds no module of it (opt_core.stock_proof)


def list_kit_files(root: str) -> List[str]:
    """Every file under ``root``, as paths relative to it (``__pycache__`` and ``.pyc``/``.pyo``/``.pyd`` excluded): presence and
    inventory, not a byte check — git (or the sdist/wheel build) already guarantees the checked-out bytes match the commit, so this
    never hashes anything."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if not f.endswith((".pyc", ".pyo", ".pyd")):
                out.append(os.path.relpath(os.path.join(dirpath, f), root))
    return out

# The kit's switches (env name -> (the default the kit's code applies when unset, where it is read, what the value does)).
KIT_SWITCHES: Dict[str, Tuple[str, str, str]] = {
    "PXD_HOIST": ("1", "hoist.py:415", "install() returns without patching when 0|false|off"),
    "PXD_HOIST_MODE": ("shape", "hoist.py:401", "shape = hoisted ops evaluated on N_sample-expanded rows like stock (bitwise target); rows = un-expanded rows (not bitwise)"),
    "PXD_HOIST_MASK": ("1", "hoist.py:433", "0|off leaves the padding-mask bias (H5) uncached"),
}


@dataclass(frozen=True)
class Lever:
    name: str
    what: str                                             # what is hoisted / changed
    cls: str                                              # forward | host | precision
    tier: str                                             # "1" = byte-identical to stock under the deterministic recipe; "2" = numerics within the engine's stated bound
    switch: str                                           # the kit switch whose value enables it (hoist levers), or the mode-table field (package levers)
    code: str                                             # file:line of the replay path / the application
    prepare: str = "hoist.py:59-182 prepare_cache"        # where the tensors are computed once per sample_diffusion call
    hit: Optional[str] = None                             # the kit's own counter name in hoist.stats()["hits"]


LEVERS: Dict[str, Lever] = {
    "h1": Lever("h1", "DiffusionConditioning pair path pair_z = LN+Linear(cat(z_trunk, relpe)) + transition_z1 + transition_z2 [N_tok^2 x 128], and the single-path base Linear(LN(cat(s_trunk, s_inputs))) [N_tok x 384] (H1s)",
                "forward", "1", "PXD_HOIST", "hoist.py:184-250 _f_forward_hoisted (DiffusionModule.f_forward)", hit="f_forward"),
    "h2": Lever("h2", "per DiffusionTransformer block (16 blocks x 16 heads) token pair bias Linear(LN(z_pair)) -> [16, N_tok, N_tok]",
                "forward", "1", "PXD_HOIST", "hoist.py:328-365 _apb_forward_hoisted (AttentionPairBias.forward, token blocks)", hit="apb_tok"),
    "h3": Lever("h3", "AtomAttentionEncoder step-invariant part: c_l, p_lm (incl. the small MLP), dense-trunk geometry [N_atom-scale]",
                "forward", "1", "PXD_HOIST", "hoist.py:184-250 _f_forward_hoisted (AtomAttentionEncoder inputs replayed from the cache)"),
    "h4": Lever("h4", "per AtomTransformer block (4 encoder + 4 decoder blocks x 4 heads) local pair bias Linear(LN(p_lm)) and the AdaLN(s) terms of the atom ConditionedTransitionBlocks",
                "forward", "1", "PXD_HOIST", "hoist.py:328-365 _apb_forward_hoisted (atom blocks), hoist.py:378-391 _ctb_forward_hoisted", hit="apb_atom"),
    "h5": Lever("h5", "_local_attention padding-mask bias (attn_bias=None branch: new_zeros + -inf fills + concat_split/unfold), 8 atom blocks per call [N_atom-scale]",
                "forward", "1", "PXD_HOIST_MASK", "hoist.py:254-294 _local_attention_hoisted (protenix.model.modules.primitives._local_attention)", prepare="hoist.py:252 _MASK_CACHE (per shape, per process)", hit="local_attn_mask"),
}

# The package's own levers, applied around upstream's InferenceRunner at the activation point (stack.py); `switch` names the ModeSpec field that selects them.
PACKAGE_LEVERS: Dict[str, Lever] = {                                # the package-lever hook (stack.py, after load_checkpoint) applies these (sizeceil.apply); modes.KIT_MODES names which mode selects which (fast: featdiet, padmask, tf32, sdedup; big: those four + rowpipe)
    "featdiet": Lever("featdiet", "the featurizer's [N_atom, N_atom] int64 bond_mask (a training-loss input no inference path reads) dropped from input_feature_dict before to_device: 8 B x N_atom^2 never reaches the GPU",
                      "host", "1", "package_levers", "sizeceil.py apply_featdiet (pxdesign.runner.inference.InferenceRunner.predict)", prepare="none (a dict pop per item)"),
    "padmask": Lever("padmask", "the atom transformer's windowed padding mask / padding bias [n_trunks, 32, 128] built from indices instead of sliced out of a dense [N_atom(+pad), N_atom(+pad)] ones / zeros tensor (4 B x N_atom^2 per construction): rearrange_qk_to_dense_trunk(compute_mask=True), rearrange_to_dense_trunk(attn_bias=None), and the h5 cache seeded so its dense construction never runs — element for element the stock values",
                     "forward", "1", "package_levers", "sizeceil.py apply_padmask (protenix.model.modules.primitives; pxd_xattempt.hoist._MASK_CACHE)", prepare="none (index arithmetic per distinct N_atom)"),
    "rowpipe": Lever("rowpipe", "the hoist kit's per-call prepare with its three pair-plane passes (H1 conditioning pair path, H2 the 16 token pair biases, the encoder pair term) evaluated in balanced row slabs (at most 256 token rows and at most 32768 (compute capability 9.0) / 65536 (8.0) token-pair rows per slab GEMM, rowpipe.slab_rows with rowpipe.gemm_rows_cap) from the integer relative-position features, each slab's rows written into preallocated outputs, and no resident pair_z (a [1, 0] shape carrier: the per-step path reads its row count only, and 1 row keeps the token transformer's per-block torch.cuda.empty_cache() off — transformer.py:447, +24 % forward at 4788 tokens with it): no [N_tok, N_tok, c] fp32 plane is materialised — 512 B x N_tok^2 resident and > 2.5 KiB x N_tok^2 transient at prepare removed; fp32-reassociation class (row-independent LayerNorm/Linear; cuBLAS may select kernels by shape)",
                     "forward", "2", "package_levers", "rowpipe.py apply_rowpipe (pxd_xattempt.hoist.prepare_cache)", prepare="rowpipe.py prepare_cache (once per sample_diffusion call, row slabs of rowpipe.slab_rows(N_tok, gemm_rows=gemm_rows_cap(device)))"),
    "tf32": Lever("tf32", "the numerics policy `tf32` (opt_core.precision.policy: matmul precision `high`, cuDNN TF32 on) set after checkpoint load: every fp32 matmul of the process — the condition embedder's and the 400-step sampler's Linear / matmul statements, all stock's — runs as TF32 tensor-core products with fp32 accumulation instead of IEEE fp32 products; the rounding of every GEMM moves (tolerance class)",
                  "precision", "2", "package_levers", "precision.py apply_tf32 (opt_core.precision.policy.apply; torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32); LEVER strategy word F4.tf32_matmul, the core's canonical id (opt_core/STRATEGIES.json)", prepare="none (two process switches)"),
    "sdedup": Lever("sdedup", "the token path's single conditioning s(t_hat) — DiffusionConditioning's single path below its step-invariant base — and everything computed from s alone (f_forward's linear_no_bias_s(layernorm_s(s)); per token block AdaLN's linear_s / linear_nobias_s in AttentionPairBias and ConditionedTransitionBlock and the two adaLN-Zero gates linear_a_last(s) / linear_s(s): six c_s->c_token projections x 24 blocks) evaluated on ONE sample row and broadcast over N_sample inside the stock elementwise products (t_hat holds one value per denoiser call, so the N_sample rows of s are identical; a call whose rows differ evaluates the stock rows, counted as fallback_by=nonuniform); exact algebra, cuBLAS kernel selection by M moves the fp32 summation order (tolerance class)",
                    "dedup", "2", "package_levers", "sdedup.py apply (opt_core.capture.hoist.RowDedup through the hoist kit's hook pxd_xattempt.fuse, hoist.py:197-205)", prepare="none (one uniformity read of t_hat [N_sample] per denoiser call)", hit="sdedup"),
}
ALL_LEVERS: Dict[str, Lever] = {**LEVERS, **PACKAGE_LEVERS}
