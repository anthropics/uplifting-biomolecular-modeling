"""The mode table — the one place a mode is defined — and how a mode resolves to the kit's switch.

The levers a mode can name (``TABLE`` below; each is one module of this package or the carried adapter): ``DEVICE_RESIDENT``
(device_resident.py: the recycle loop's operands kept on the device, the float16 outputs fetched once — placement only), ``SUBBATCH``
(subbatch.py: ``global_config.subbatch_size`` raised against the visible device), ``TRIMUL_PALLAS`` (trimul_pallas.py: the fused
triangle multiplication through the shared core's provider face by the mode's tier word), ``AF_PALLAS_ATTN`` (the carried adapter ``opt/forward/af2_pallas_flash/af2_pallas_flash/af2_pallas_attn.py``
over the shared core's flash-attention kernel: ``enable()`` :69-103, its switch ``AF_PALLAS_ATTN=1`` read at import :124-125,
``AF_PALLAS_ATTN_ALL`` :77 left at its default "0"; bf16 attention arithmetic re-associated — never bit-identical to stock, deterministic run
to run, af2_pallas_flash/README.md:14-16), ``PALLAS_MSA`` (msa_attn.py: MSA-column and extra-MSA row attention through the same kernel), ``MSA_COL_CUDNN``
(msa_col_cudnn.py: MSA column attention through cuDNN's fused flash attention with key lengths), ``ROWPAIR`` (big.py: the row-sharded pair representation over
``--n_gpu P`` > 1 GPUs), ``TEMPL_DEDUP`` (templ_dedup.py: a template row equal to its
predecessor reuses that row's embedding inside the multimer template scan — colabfold's four mock rows are embedded once), ``TRANSITION``
(transition.py: AlphaFold's Transition module through the provider's transition face by the mode's tier word) and ``TRIATTN_XLA`` (triattn_xla.py: the shared core provider's
pre-compiled triangle-attention row ``opt_core.kernels.triattn_xla`` inside ``AF_PALLAS_ATTN``'s binding of the pair-biased attention sites —
triangle attention starting / ending node, MSA row attention — which calls the provider ``opt_core.kernels.pallas`` by the mode's tier word, ``TIER_WORDS``).

Modes (``MODES``; ``DEFAULT_MODE`` = fast — the package default when ``--mode`` is omitted: fast wherever a fast composition ships at the
default settings; ``COLABFOLD_OPT`` unset in a caller's environment = off, nothing activates):
  off     stock: no environment variable set, nothing of the kit applied, ``colabfold_batch`` as shipped.
  exact   ``DEVICE_RESIDENT`` — placement only; Tier 1 (byte-identical outputs expected: the same executables on the same arguments).
  fast    ``DEVICE_RESIDENT``, ``SUBBATCH``, ``TRIMUL_PALLAS``, ``AF_PALLAS_ATTN``, ``PALLAS_MSA``, ``TRIATTN_XLA``, ``MSA_COL_CUDNN``, ``TEMPL_DEDUP``, ``TRANSITION`` in the process that runs
          ``colabfold.batch.run`` (stack.py). Tier 2 (numerics differ from stock at the bf16 rounding level, af2_pallas_flash/README.md:14).
  big   fast's set plus ``ROWPAIR``, installed at ``--n_gpu P`` > 1 only (P = 1: fast's bytes). Tier 2. At ``--n_gpu 8`` `pred` starts the
          model process with jax's memory-pool fraction 0.90 (``N_GPU_MEM_FRACTION``) unless the caller exported a different fraction: any value other than the image preset 0.95 is kept.
The deployment lever ``XLA_CACHE`` (xla_cache.py) is placed in every mode of the kit and is in no mode's lever tuple (no numerics).
An unknown name: ``resolve`` raises UnsupportedMode by name.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

ENV = "COLABFOLD_OPT"                                    # the package's switch: COLABFOLD_OPT=<mode>
MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
BIG = "big"                                          # the memory mode: fast's lever set + the row-sharded pair stack at --n_gpu P>1 (big.py)
DEFAULT_MODE = "fast"                                    # the package default when --mode is omitted: fast wherever a fast composition ships
KIT_MODE_NAMES: Tuple[str, ...] = ("exact", "fast", "big")   # the modes that activate a lever
TIER_WORDS: Dict[str, str] = {"exact": "exact", "fast": "fast", "big": "big"}   # mode -> the tier word its levers bind on the shared core's provider (opt_core.kernels.pallas: the provider's cell table names the row served per call class for the word; exact binds no attention site)
KIT_SWITCH = "AF_PALLAS_ATTN"                            # the carried kit's own switch (af2_pallas_attn.py:124)
KIT_SWITCH_ON = "1"
KIT_MODULE = "af2_pallas_attn"                   # the carried kit's integration module (af2_pallas_flash/af2_pallas_attn.py)
KIT_PACKAGE = "af2_pallas_flash"                         # the carried kernel package (the LEVER lines' impl= / kernel= token)
LEVER = "AF_PALLAS_ATTN"                                 # the kernel lever's name in the activation line and the report
MSA_LEVER = "PALLAS_MSA"                                # MSA-column + extra-MSA row attention through the flash kernel (msa_attn.py): opt_core.kernels.pallas_attn_serve
MSA_LEVER_MODULE = "colabfold_opt.msa_attn"
TRIMUL_LEVER = "TRIMUL_PALLAS"                           # the fused triangle multiplication through the shared core's provider face by the mode's tier word (trimul_pallas.py over opt_core.kernels.pallas.serve)
TRIMUL_LEVER_MODULE = "colabfold_opt.trimul_pallas"      # its module
COL_LEVER = "MSA_COL_CUDNN"                              # MSA column attention through cuDNN's fused flash attention with key lengths (msa_col_cudnn.py; jax.nn.dot_product_attention implementation=cudnn)
COL_LEVER_MODULE = "colabfold_opt.msa_col_cudnn"
TRIATTN_LEVER = "TRIATTN_XLA"                            # the pair-biased attention sites (triangle start/end, MSA row) on the shared core's pre-compiled triangle-attention kernels
TRIATTN_LEVER_MODULE = "colabfold_opt.triattn_xla"      # (triattn_xla.py over opt_core.kernels.triattn_xla: sm_90a CUDA / K2B AOT cubins through one XLA-FFI launcher), over AF_PALLAS_ATTN's class
TEMPL_LEVER = "TEMPL_DEDUP"                              # identical template rows embedded once (templ_dedup.py: the multimer template scan reuses the previous row's embedding for an equal row)
TEMPL_LEVER_MODULE = "colabfold_opt.templ_dedup"
TRANSITION_LEVER = "TRANSITION"                          # AlphaFold's Transition module (pair / MSA / extra-MSA / template pair stack) on the shared core's provider face by the mode's tier word (transition.py: opt_core.kernels.pallas.serve.transition)
TRANSITION_LEVER_MODULE = "colabfold_opt.transition"
PAIR_MUL_LEVERS: Tuple[str, ...] = (TRIMUL_LEVER,)      # the one-device TriangleMultiplication lever(s): under big at --n_gpu P>1 the row-sharded pair stack owns the class and the activation drops it (state=off reason=n_gpu>1)
ONE_DEVICE_PAIR_LEVERS: Tuple[str, ...] = (TRIMUL_LEVER, TEMPL_LEVER, TRANSITION_LEVER)   # dropped at --n_gpu P > 1 (state=off reason=n_gpu>1): TRIMUL_PALLAS (the row-sharded stack owns TriangleMultiplication: full planes), TEMPL_DEDUP (the row-sharded template region owns multimer TemplateEmbedding.__call__), TRANSITION (the recipe's transition body). TRIATTN_XLA and MSA_COL_CUDNN stay on at P > 1 — both bind over AF_PALLAS_ATTN's / PALLAS_MSA's rebound Attention class, which the recipe's row-block bodies instantiate, and serve there with the one-GPU shapes (triangle attention: this GPU's row chunks x N keys behind the gathered bias; MSA column attention: the replicated MSA stack)
ONE_DEVICE_LEVERS: Tuple[Tuple[str, str], ...] = ((TRIMUL_LEVER, TRIMUL_LEVER_MODULE), (TEMPL_LEVER, TEMPL_LEVER_MODULE), (TRANSITION_LEVER, TRANSITION_LEVER_MODULE))   # (lever, module) of ONE_DEVICE_PAIR_LEVERS: the activation drops each at --n_gpu P>1 with the module's N_GPU_REASON (stack.activate)
SUPERSEDES: Dict[str, Tuple[str, ...]] = {TRIATTN_LEVER: (LEVER,), COL_LEVER: (MSA_LEVER,)}   # (+ MSA_COL_CUDNN takes the MSA-column site from PALLAS_MSA, which keeps the extra-MSA rows) a lever bound OVER another's class that serves that lever's calls itself: TRIATTN_XLA takes the pair-biased
                                     # 32-channel sites from AF_PALLAS_ATTN from its size floor up. When the superseded lever ends a run with NOTHING having reached it
                                     # (calls=0 — the single-chain route without templates from its size floor: no 16-channel template head is left for it) while
                                     # the superseding lever served calls (its documented bias-free fallbacks aside), it was superseded BY NAME (`superseded_by=TRIATTN_XLA:<n>` on its LEVER line; manifest.superseded_by):
                                     # by design, not a partial activation. calls=0 with no superseding call served stays PARTIAL.
SUBBATCH_LEVER = "SUBBATCH"                              # the sub-batch lever (subbatch.py): global_config.subbatch_size decided by opt_core.jax_design.subbatch_policy
SUBBATCH_LEVER_MODULE = "colabfold_opt.subbatch"
HOST_LEVER = "DEVICE_RESIDENT"                           # the package's own transfer lever (device_resident.py): placement only, in exact and fast
HOST_LEVER_MODULE = "colabfold_opt.device_resident"
TP_LEVER = "ROWPAIR"                                     # the row-sharded pair stack of big at --n_gpu P>1 (big.py; opt_core.mem.rowpair_jax)
TP_LEVER_MODULE = "colabfold_opt.big"
ENV_N_GPU = ENV + "_N_GPU"                                # the model process's copy of --n_gpu (cli.pred exports it for big; absent = 1)
N_GPU_SUPPORTED: Tuple[int, ...] = (1, 2, 4, 8)          # the P set this kit ships (refused by name outside it; 8 = one eight-GPU host)
TP_MODES: Tuple[str, ...] = ("big",)                   # --n_gpu > 1 is refused by name under every other mode (opt_core.mem.ngpu)
MEM_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"        # jax's client memory-pool fraction; the image presets 0.95 (environment/Dockerfile ENV = stock/PINS.json image.env)
MEM_FRACTION_ENV_ALT = "XLA_CLIENT_MEM_FRACTION"          # the other name jax reads for the same setting (a caller who set it chose a fraction)
N_GPU_MEM_FRACTION: Dict[int, str] = {8: "0.90"}          # big --n_gpu P -> the pool fraction `pred` starts the model process with (cli.n_gpu_mem_fraction): at P = 8 NCCL's
                                                          # communicators, which allocate OUTSIDE jax's pool, need the headroom — the P = 8 line runs at 0.90; a P absent here
                                                          # (1, 2, 4) keeps the environment's value (the image's 0.95); a caller's own fraction — any value other than the image preset — is kept
NO_DET_REASON = ("the package carries no deterministic recipe: at production numerics stock ColabFold and every mode can differ run to run at the same seed "
                 "(a complex with more than one plausible chain arrangement flips between arrangements), and bitwise run-to-run reproducibility, stock's own "
                 "and `exact` = stock alike, holds only under XLA's deterministic flags, XLA_FLAGS=\"--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true\" "
                 "in the process environment, which the caller sets and this package never does")

# mode -> (levers in application order, the carried kit's environment, the numerics tier the mode promises)
TABLE: Dict[str, Tuple[Tuple[str, ...], Dict[str, str], object]] = {
    "off":   ((), {}, "stock"),
    "exact": ((HOST_LEVER,), {}, 1),                                     # byte-identical outputs expected (placement only)
    "fast":  ((HOST_LEVER, SUBBATCH_LEVER, TRIMUL_LEVER, LEVER, MSA_LEVER, TRIATTN_LEVER, COL_LEVER, TEMPL_LEVER, TRANSITION_LEVER), {KIT_SWITCH: KIT_SWITCH_ON}, 2),     # + the sub-batch raise (a different XLA program) + the fused triangle multiplication + the flash-attention kernel (bf16 re-association) + the pre-compiled triangle-attention kernels over it + cuDNN on the MSA column + the template de-duplication + the fused transition: tier 2
    "big": ((HOST_LEVER, SUBBATCH_LEVER, TRIMUL_LEVER, LEVER, MSA_LEVER, TRIATTN_LEVER, COL_LEVER, TEMPL_LEVER, TRANSITION_LEVER, TP_LEVER), {KIT_SWITCH: KIT_SWITCH_ON}, 2),   # fast's set + the row-sharded pair stack (installed at --n_gpu P>1 only, where it owns the pair sites and the one-device pair levers TRIMUL_PALLAS / TEMPL_DEDUP / TRANSITION are dropped, ONE_DEVICE_PAIR_LEVERS; P=1 = fast's bytes)
}


class UnsupportedMode(ValueError):
    """A name that is no mode of the table."""


def resolve(mode: str) -> Dict[str, object]:
    """The mode's row: its levers (application order), the carried kit's environment it exports and its tier. Raises UnsupportedMode for
    any name outside MODES."""
    if mode not in MODES:
        raise UnsupportedMode(f"{mode!r} is not a mode ({'|'.join(MODES)}; opt/colabfold_opt/modes.py)")
    levers, kit_env, tier = TABLE[mode]
    return {"mode": mode, "levers": tuple(levers), "kit_env": dict(kit_env), "tier": tier}


def tier_word(mode: str) -> str:
    """The provider tier word a kit mode binds (TIER_WORDS): ``fast`` -> fast, ``big`` -> big, ``exact`` -> exact."""
    return TIER_WORDS[mode]


def from_env(environ=None) -> str:
    """The mode a caller's environment names: COLABFOLD_OPT, else `off` (unset activates nothing)."""
    import os
    environ = os.environ if environ is None else environ
    return (environ.get(ENV) or "off").strip().lower() or "off"


def n_gpu_from_env(environ=None) -> int:
    """--n_gpu as the model process sees it: COLABFOLD_OPT_N_GPU (cli.pred exports it under big), absent = 1. A value that is not a
    positive integer raises ValueError (the activation names it)."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV_N_GPU) or "").strip()
    if not raw:
        return 1
    p = int(raw)
    if p < 1 or str(p) != raw:
        raise ValueError(f"{ENV_N_GPU}={raw!r} is not a positive integer")
    return p
