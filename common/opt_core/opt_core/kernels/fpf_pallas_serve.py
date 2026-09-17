"""Generic serving layer for the carried ``fpf_pallas`` kernels (``META/fpf_pallas.json``; JAX, Pallas-Triton lowering, inference only) and
the carried ``pallas_glut`` kernel (``META/pallas_glut.json``): the fused pair-stack blocks of a pairformer / evoformer as
PURE FUNCTIONS over arrays and a dict of parameters, the per-GPU tile tables, and the served-shape predicate as one function that
answers with a refusal NAME. Nothing here names an engine or a framework module: a kit keeps only its adapter (the Haiku / Flax module
subclass that reads the module's own parameters, asks ``served_reason``, and calls a block here or falls back to its stock body BY NAME).
Stdlib at import; jax and the kernels are imported inside the calls (``kernels()``).

Blocks (``act`` = pair activation ``[N, N, C]``; masks ``[N, N]``; every block returns ``[N, N, C]`` in ``act.dtype``):

    trimul_block(act, mask, params, *, equation, cfg=None, cc=None, precision="std")
        Triangle multiplication, ``equation`` = ``"outgoing"`` | ``"incoming"`` (or the einsum strings ``"ikc,jkc->ijc"`` | ``"kjc,kic->ijc"``).
        Three launches: Pallas prologue (input LayerNorm + GLU + mask, written directly as channel-major planes), the cubic contraction
        (XLA/cuBLAS batched GEMM ``cik,cjk->cij``; ``cfg["ein"]="pallas"`` selects the carried Pallas GEMM), Pallas epilogue (centre LayerNorm +
        output projection + sigmoid gate, the input LayerNorm recomputed rather than re-read).
        ``params`` (bias-free Linears): ``ln_in_scale[C]``, ``ln_in_offset[C]``; ``w_proj[C,2C]``, ``w_gate[C,2C]``
        (GLU weights, INTERLEAVED columns: column ``2c`` feeds left-plane channel ``c``, column ``2c+1`` right-plane channel ``c`` — the layout of a
        module whose stock body does ``einsum('...ij->ji...')``-free ``prj[0::2] / prj[1::2]`` after the transpose; ``trimul_params()`` builds it
        from separate left/right matrices); ``ln_c_scale[C]``, ``ln_c_offset[C]``; ``w_out[C,C]``; ``w_gl[C,C]`` (gate = ``sigmoid(LN_in(act) @ w_gl)``).
    tri_attn_block(act, pair_mask, params, *, orientation, ending_bias_transposed=True, key_mask="by_column", cfg=None, cc=None, precision="std")
        Triangle self-attention, ``orientation`` = ``"starting"`` | ``"ending"`` (per-row | per-column; ``"per_row"`` / ``"per_column"`` accepted).
        Three launches: ONE Pallas prologue (LayerNorm + q|k|v|pair-bias projections; the ending node READS ``act`` transposed inside the kernel,
        no XLA transpose pass), the flash core (online softmax in the exp2 domain, f32 accumulation, finite negative for masked keys: a fully
        masked row gives the uniform average exactly as a stock softmax, never NaN), ONE Pallas epilogue (LayerNorm recomputed + sigmoid gate +
        output projection; transposed writes for the ending node). ``ending_bias_transposed``: for the ending node the pair bias is the
        projection of the TRANSPOSED normalised activation (= a module that transposes ``act`` before everything, or a checkpoint
        whose ending-node pair bias is stored transposed: ``True``); ``False`` = the bias of the un-transposed activation.
        ``params`` (kernel layout; build it with ``attn_params()``): ``ln_scale[C]``, ``ln_offset[C]`` (f32), ``wq_t[C,HD]``, ``wk_t[C,HD]``,
        ``wv2[C,HD]``, ``wb16[C,16]`` (pair-bias projection zero-padded to 16 columns), ``wg_t[C,HD]``, ``wo[HD,C]`` (bf16), ``H``, ``D`` (ints).
        ``key_mask``: ``"by_column"`` = key ``k`` of batch line ``b`` is valid iff ``pair_mask[k, b]`` (both orientations);
        ``"by_line"`` = the attended line's own mask entry (rows for the starting node, columns for the ending node).
    attention_core(q, k, v, bias, key_mask, *, cfg=None, cc=None)
        The flash core alone: ``q/k/v [B,S,H,D]``, ``bias [H,S,S]`` (one pair bias shared over the batch of rows), ``key_mask [B,S]`` bool
        -> ``[B,S,H,D]``; softmax scale ``1/sqrt(D)`` applied inside. ``S % bq == 0 and S % bk == 0``.
    glu_transposed_masked(x, weights, mask2d, *, activation, config=None)
        The carried ``pallas_glut`` kernel: ``== transpose(tokamax.gated_linear_unit(x, weights, activation), (2,0,1)) * mask2d[None]`` BIT-EXACT
        (same tile policy, same rounding points); ``x [A,B,K]``, ``weights [K,2,N]`` (``[:,0]`` gate, ``[:,1]`` projection), ``mask2d [A,B]`` ->
        ``[N,A,B]``. Importing it imports ``tokamax``; the other functions here never do.

Parameter builders (pure layout conversions, tiny XLA ops fused at trace time):
    attn_params(*, ln_scale, ln_offset, q_w, k_w, v_w, bias_w, gate_w, out_w)      math layout: ``q_w/k_w/v_w/gate_w [C,H,D]``, ``bias_w [C,H]``,
        ``out_w [H,D,C]`` -> the kernel dict above.  ``attn_params_from_tree(tree)`` = the carried converter for a parameter tree in the
        bias-free module layout (``q_projection (H,D,C)``, ``k_projection (H,D,C)``, ``v_projection (C,H,D)``, ``gating_query (HD,C)``,
        ``output_projection (HD,C)``, ``pair_bias_projection (C,H)``, ``act_norm {scale, offset}``).
    trimul_params(*, ln_in_scale, ln_in_offset, left_w, right_w, left_gate_w, right_gate_w, ln_c_scale, ln_c_offset, out_w, gate_w)
        separate ``[C,C]`` matrices -> the interleaved ``w_proj/w_gate [C,2C]`` dict.

Served shapes and dtypes — ``served_reason(kind, act_shape, act_dtype, mask_shape, num_head=None)`` -> ``None`` when served, else the NAME of
the refusal (``rank_not_3`` · ``dtype_not_served`` (an activation dtype other than bfloat16 | float32) · ``not_square`` · ``n_not_tile_multiple`` (N not a multiple of the tiles:
64 on the H100 table, ``required_multiple(kind, n, cc)``; answered only with ``pad=False`` — by default the blocks serve ANY ``N`` by
zero-padding to ``kernel_n(kind, N, cc)`` and slicing back, which is exact) · ``c_not_multiple_of_16`` · ``c_out_of_range`` ·
``mask_shape`` · ``head_dim_unsupported`` · ``heads_x_dim_gt_256``); a kit's adapter falls back to its stock body under that name and COUNTS it
(a fallback is a row-level fact, never silent). DTYPE (``SERVED_DTYPES[kind]``): both
blocks serve ``bfloat16`` AND ``float32`` activations, by two kernel sets. The carried bf16 kernels round LayerNorm outputs, projections,
q/k/v and the softmax probabilities to bf16 at a bf16 stock model's own rounding points and run every MMA as bf16 x bf16 -> f32; handing
them f32 activations would be a silent downcast, so f32 activations never reach them: ``trimul_block`` and ``tri_attn_block`` route f32
activations to the f32 kernels of ``fpf_pallas_f32`` (same launches, nothing rounded to bf16 anywhere, f32 x f32 -> f32 MMAs at a NAMED
operand precision REQUIRED on every f32 call — ``precision="tf32"`` (tensor-core tf32 operands rounded to nearest, f32 accumulate — the class of
an XLA f32 GEMM at default precision on sm_80+, i.e. what a stock f32 model runs on those parts; not f32-exact; the fast word), ``"tf32x3"`` (3-pass
tf32 via the dot-algorithm preset, ~f32 accuracy; raises on a jax without the preset), ``"ieee"`` (CUDA-core f32 FMA: f32-exact, 45-60x the tf32 time —
validation only); ``"std"`` / no word is the refusal ``precision_required``; the NUMERICS CLASS the word lands in depends on the jax
release line (``F32_LINES``: Pallas-Triton rounds f32 operands to nearest tf32 on jax >= 0.10 — the class of the XLA-default f32 body — and
truncates them on jax 0.5 — the bf16-operand class over a block; the attention's ``'ieee'`` cannot launch on jax 0.5 and is refused there by
name, ``precision_unlaunchable``) — a kit prints ``f32_class=<f32_class(precision)>`` beside ``mma=<precision>`` on its SERVED line;
the triangle multiplication's cubic contraction stays ONE XLA batched GEMM (default precision at ``"tf32"``, HIGHEST at the other two words);
the kit names the word on its SERVED line as ``mma=<precision>``). The f32 route reads f32
parameters: build them with ``attn_params(..., weights_dtype=jnp.float32)`` / ``trimul_params`` on f32 arrays (bf16 parameters are read
exactly but carry their own rounding). Its tile rows (``trimul_f32`` / ``trimul_f32_by_n``, ``attn_f32_default`` / ``attn_f32_by_n``) are
their own sweep, so ``required_multiple`` / ``kernel_n`` / ``trimul_cfg`` / ``attn_cfg`` take ``dtype=``. ``attention_core`` alone accepts f32 ``q/k/v`` on the carried core (Triton's default f32 input precision, tf32-class). BIASES: the carried kernels have no Linear biases; a parameter dict that carries the bias
arrays (``b_proj/b_gate/b_out/b_gl`` on the triangle multiplication, ``bg/bo`` on the attention — the biased parameterisation;
``trimul_params`` / ``attn_params`` build them) routes the SAME call to the ``+bias`` variants in ``fpf_pallas_bias`` (one FMA per bias
inside the same prologue / epilogue kernels; everything else is the carried kernel's); some-but-not-all bias keys is the refusal
``bias_keys_incomplete`` — never a silent zero.

Tile tables — ``TILE_TABLES[cc]``: the tile configurations per CUDA compute capability (``"8.0"`` A100, ``"9.0"`` H100, ``"10.0"`` B200,
``"10.3"`` B300), each ``{trimul, trimul_by_n, attn_default, attn_by_n}`` plus, where swept, ``{attn_f32_default, attn_f32_by_n}`` (the f32 attention
kernels: f32 tiles hold twice the bytes, so their rows are their own sweep) and, where the jax lines' Pallas-Triton lowerings want different
f32 tiles on the same part (sm_80), ``f32_by_jax_line`` = {<jax line>: that line's f32 rows}; ``tables_for(cc)`` -> ``(cc, table, own)`` (the
table as served on the importable jax's line); a capability without
a table of its own is REFUSED by name (``no_tiles``, detail ``no-tiles:<cc>``) unless ``OPT_CORE_PALLAS_ALLOW_FALLBACK_CC=1`` opts in to the
``"9.0"`` table, which the caller then records as ``tiles=fallback:<cc>->9.0`` (``tiles_label``); ``trimul_cfg(n, cc, dtype=)`` / ``attn_cfg(n,
cc, dtype=)`` -> the merged dict for pair size ``n`` (``dtype="float32"`` reads the f32 rows ``{trimul_f32, trimul_f32_by_n}``; a table without
them refuses by name: ``no_tiles``, detail ``no-f32-tiles:<cc>``). A kit passes an explicit ``cfg=`` override (its own sweep) or nothing; it never keeps a silent local table.

Numerics class: not bit-exact to an XLA stock body (bf16 re-association: one rounding per fused kernel where stock rounds after every op);
error against an f64 reference is at or below the stock bf16 body's (the carried ``triangle_multiplication_reference`` /
``grid_self_attention_reference`` are the op-by-op replicas a parity test compares both against). Run-to-run deterministic (no atomics).
``precision="hi"`` keeps the trimul contraction result / the attention output in f32 between kernels (MMAs stay bf16): measured no closer to
stock and 2-3 % slower; ``"std"`` is the setting. On the f32 routes ``precision`` names the MMA operand precision instead
(``F32_PRECISIONS`` = ``fpf_pallas_f32.PRECISIONS``: ``"tf32"`` | ``"tf32x3"`` | ``"ieee"``, REQUIRED — ``"std"`` / ``None`` is
``precision_required``, anything else ``unknown_precision``); at ``tf32`` its class against an f32 reference at HIGHEST precision is the tf32 class (rel-rms 2-7e-4, at or below an
XLA f32 body at default precision on the same part on every tested shape); at ``ieee`` 2-3e-7; run-to-run bit-exact.
"""
import importlib
import re
import sys
from typing import Any, Dict, Optional, Sequence, Tuple

NAME = "fpf_pallas"                                   # the carried-kernel name (META/fpf_pallas.json)
GLUT_NAME = "pallas_glut"                             # the carried GLU kernel (META/pallas_glut.json)
TRIMUL_FILE, TRIATTN_FILE = "trimul_pallas", "triattn_pallas"

# Tile tables per GPU architecture (CUDA compute capability). Re-tune on another part with the kernels' parity/timing launcher and
# add a row; a capability without a row but of a named generation is served GENERATION_SAFE_TABLES (tables_for); any other is refused by name
# unless OPT_CORE_PALLAS_ALLOW_FALLBACK_CC=1.
TILE_TABLES: Dict[str, Dict[str, Any]] = {
    "8.0": dict(   # A100 80GB (sm_80: 163 KB of shared memory per block, mma.sync — no TMA/wgmma), swept on the jax 0.10.2 and jax 0.5.3 images (A100-SXM4 and
                   # A100 PCIe parts); every default-row tile divides 64 (pair sizes pad to the same multiple as on sm_90). The sm_90 rows also compile and hold
                   # the class here; these are 1.1x (bf16 trimul), 1.1-1.4x (bf16 attention) and 2-5x (f32 attention) faster than them on this part.
        trimul=dict(t1=64, w1=4, s1=2, t2=64, w2=4, s2=2, ein="xla"), trimul_by_n={},
        attn_default=dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=32, wa=4, sa=3),
        attn_by_n={768: dict(bq=128), 1024: dict(t2=128, w2=8, bk=64), 1536: dict(bq=128), 2048: dict(t1=128, bk=64)},
        # f32 rows (fpf_pallas_f32): one attention row serves both jax lines; the triangle multiplication's fastest tiles differ per jax line on
        # sm_80 (each line's row is ~1.25x slower on the other): the base rows are the jax 0.10 line's, f32_by_jax_line["0.5"] the jax 0.5 line's.
        attn_f32_default=dict(t1=64, w1=4, hc1=64, t2=32, w2=4, hc2=32, bq=64, bk=64, wa=4, sa=2, vt=1),
        attn_f32_by_n={},
        trimul_f32=dict(t1=32, w1=4, hc1=64, t2=64, w2=4, hc2=32, ein="xla"),
        trimul_f32_by_n={},
        f32_by_jax_line={"0.5": dict(trimul_f32=dict(t1=64, w1=4, hc1=32, t2=64, w2=4, hc2=16, ein="xla"), trimul_f32_by_n={})}),
    "9.0": dict(   # H100 80GB HBM3
        trimul=dict(t1=64, w1=8, s1=2, t2=64, w2=8, s2=2, ein="xla"), trimul_by_n={},
        attn_default=dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=64, wa=4, sa=3),
        attn_by_n={1024: dict(bq=128, bk=32, wa=4, sa=3)},
        attn_f32_default=dict(t1=64, w1=4, hc1=128, t2=64, w2=4, hc2=64, bq=64, bk=64, wa=4, sa=2, vt=1),   # the f32 attention kernels (fpf_pallas_f32); every tile divides 64
        attn_f32_by_n={1408: dict(bq=128, bk=32, sa=2)},
        trimul_f32=dict(t1=64, w1=4, hc1=32, t2=64, w2=4, hc2=32, ein="xla"),                             # the f32 triangle-multiplication kernels (fpf_pallas_f32)
        trimul_f32_by_n={}),
    "10.0": dict(  # B200 (sm_100a), swept on jax 0.10.2 / CUDA 12.9 wheels
        trimul=dict(t1=128, w1=8, s1=2, t2=64, w2=4, s2=3, ein="xla"),
        trimul_by_n={256: dict(t2=128, w2=8, s2=2), 768: dict(t2=128, w2=8, s2=2)},
        attn_default=dict(t1=128, w1=4, t2=128, w2=4, bq=128, bk=32, wa=4, sa=2),
        attn_by_n={256: dict(sa=3)}),
    "10.3": dict(  # B300 SXM6 AC (sm_103a): module-level sweeps on this part were noisy (two sweeps disagree); rows >= 512 use the clean B200
                   # configs (the B300 re-sweep picked the same t1=128/w1=8 family; in-model 1.246/1.261/1.253x at 512/768/1024), N=256 uses the
                   # H100 configs (in-model 1.060x on B300; the B300-swept 256 config measured 0.983x).
        trimul=dict(t1=128, w1=8, s1=2, t2=64, w2=4, s2=3, ein="xla"),
        trimul_by_n={256: dict(t1=64, w1=8, s1=2, t2=64, w2=8, s2=2), 768: dict(t2=128, w2=8, s2=2)},
        attn_default=dict(t1=128, w1=4, t2=128, w2=4, bq=128, bk=32, wa=4, sa=2),
        attn_by_n={256: dict(t1=64, w1=4, t2=64, w2=4, bq=64, bk=64, wa=4, sa=3)}),
}
TILE_TABLES_SOURCE = {"8.0": "A100 sweeps", "9.0": "H100 sweeps", "10.0": "B200 sweeps", "10.3": "B300 sweeps"}
FALLBACK_CC = "9.0"
FALLBACK_ENV = "OPT_CORE_PALLAS_ALLOW_FALLBACK_CC"           # "1" = a compute capability without a table of its own runs the FALLBACK_CC table, RECORDED

# Generation SAFE rows: a compute capability WITHOUT a table of its own but of a generation named here (``generation(cc)``: "8.6" -> "8.x") is
# served these rows — the smallest tiles every kernel supports (32-row tiles, one pipeline stage, hidden chunks of 32 / 16; the largest single
# MMA working set is <= 56 KB at C=128, under the 99 KB of shared memory an sm_86 / sm_89 program may use), valid for ANY pair size the
# kernels serve (every tile divides 64: required_multiple / kernel_n pad as on the tuned tables). The lever counts as ENGAGED on them:
# tables_for answers ``own=False``, ``tiles_label`` says ``safe:<generation>`` and the shared helper (kernels/safe_settings.py) prints its ONE
# ``safe settings served`` line and carries the census word ``settings=safe:no_cell:tile_table`` (safe_net().word()); the tuned rows
# of a part with its own table are untouched. A capability of no named generation (below 8.0, unparsable, none) stays the refusal ``no_tiles``.
SAFE_TABLE: Dict[str, Any] = dict(
    trimul=dict(t1=32, w1=4, s1=1, t2=32, w2=4, s2=1, ein="xla"), trimul_by_n={},
    attn_default=dict(t1=32, w1=4, t2=32, w2=4, bq=32, bk=32, wa=4, sa=1), attn_by_n={},
    trimul_f32=dict(t1=32, w1=4, hc1=32, t2=32, w2=4, hc2=16, ein="xla"), trimul_f32_by_n={},
    attn_f32_default=dict(t1=32, w1=4, hc1=32, t2=32, w2=4, hc2=32, bq=32, bk=32, wa=4, sa=1, vt=1), attn_f32_by_n={})
GENERATION_SAFE_TABLES: Dict[str, Dict[str, Any]] = {"8.x": SAFE_TABLE, "9.x": SAFE_TABLE, "10.x": SAFE_TABLE}
SAFE_SHAPE_WORD = "tile_table"                                 # the helper's no_cell shape word when these rows serve: settings=safe:no_cell:tile_table
_SAFE_ANNOUNCED: Dict[str, str] = {}                           # cc -> generation served in this process

EQUATIONS = {"outgoing": "ikc,jkc->ijc", "incoming": "kjc,kic->ijc", "ikc,jkc->ijc": "ikc,jkc->ijc", "kjc,kic->ijc": "kjc,kic->ijc"}
ORIENTATIONS = {"starting": False, "ending": True, "per_row": False, "per_column": True}     # -> the kernels' `transpose` flag
HEAD_DIMS = (16, 32, 64)                              # head dims the attention prologue/epilogue tile (D a power of two >= 16)
MAX_HD = 256                                          # H * D bound (one program holds the [t, HD] q/k/v tiles)
N_MULTIPLE = 64                                       # pair-size granularity of the "9.0" table (every tile in it divides 64); other tables: required_multiple(kind, cc)
C_RANGE = (16, 256)                                   # pair channels: C % 16 == 0 within this range
SERVED_DTYPES = {"trimul": ("bfloat16", "float32"), "triattn": ("bfloat16", "float32")}   # per block kind; float32 = the fpf_pallas_f32 kernels
SERVED_DTYPE = "bfloat16"                             # the carried kernels' dtype (every kind)
F32_DTYPE = "float32"
# F32_PRECISIONS (module attribute, served lazily below) = fpf_pallas_f32.PRECISIONS: the MMA operand precision words of the f32 route,
# stated once in the kernel module; the word is REQUIRED on the f32 route ("std" / None -> the refusal precision_required).

# refusal names (served_reason / Refusal.kind)
NO_TILES = "no_tiles"                                       # detail "no-tiles:<cc>": no tile table for this compute capability
RANK_NOT_3, NOT_SQUARE = "rank_not_3", "not_square"
DTYPE_NOT_SERVED = "dtype_not_served"                       # an activation dtype outside SERVED_DTYPES[kind] (bfloat16 | float32)
DTYPE_NOT_BF16 = DTYPE_NOT_SERVED                           # alias kept for one release for callers that imported the older name; served_reason answers dtype_not_served
UNKNOWN_PRECISION = "unknown_precision"
PRECISION_REQUIRED = "precision_required"                   # the f32 route names its MMA class on every call (tf32 | tf32x3 | ieee); no default word
PRECISION_UNLAUNCHABLE = "precision_unlaunchable"           # the word names a kernel that cannot launch on this jax line (F32_LINES[line]["unlaunchable"])

# The f32 route's NUMERICS CLASS per jax release line (jax major.minor -> row), stated by name so a kit prints it beside mma=<precision> on its
# SERVED line (`f32_class(precision)`): the Pallas-Triton lowering of an f32 `dot` differs between jax lines, so the same kernel bytes at
# precision 'tf32' land in the class of the XLA-default f32 body on one line and in the bf16-operand class on another.  Words:
#   'body'    same class as the pure-XLA f32 body at DEFAULT precision: rel-rms vs a HIGHEST reference <= ratio x the body's (the core rule);
#   'bf16op'  bf16-operand class: rel-rms vs the HIGHEST reference under the row's measured `ceiling[kind]`, above the body's own error;
#   'f32'     the f32 class PER CELL ('tf32x3' / 'ieee': <= 5e-5 vs the HIGHEST reference) — a per-cell class says nothing monotone about an
#             end-to-end model (a float32 model at attention 'tf32x3' moved FURTHER from its stock run-to-run band than at 'tf32'): that is kit evidence;
#   'unlisted:<line>'  no row for this jax line: the class is unmeasured there (the core's in-class test fails by name until a row exists).
# `unlaunchable` names precision words whose kernel cannot launch on that line: `f32_precision` refuses them BY NAME (precision_unlaunchable)
# instead of an XlaRuntimeError mid-model.  Rows are evidence, added only from a measured run (the `evidence` field).
F32_LINES: Dict[str, Dict[str, Any]] = {
    "0.5": {"tf32_class": "bf16op",                                                   # this line's Pallas-Triton converts f32 operands to tf32 by TRUNCATION
            "cause": "Pallas-Triton f32 dot at DEFAULT/tf32 = truncated tf32 operands: per-dot rel-rms 7.7e-4 vs XLA's round-to-nearest tf32 2.9e-4 "
                     "(1024^2 probe, H100); over a block the truncation compounds into the bf16-operand class",
            "ceiling": {"triattn": 6.0e-3, "trimul": 1.5e-3},                       # rel-rms vs HIGHEST at 'tf32'; measured 4.3-4.5e-3 / 1.06e-3 (H100, C=128, N 256-512)
            "unlaunchable": {("triattn", "ieee"): "the f32 attention prologue at Precision.HIGHEST requests 98304 B of shared memory per program on this "
                                                  "line's Triton (a bare 128x128x128 HIGHEST dot: 131072 B): CUDA_ERROR_OUT_OF_MEMORY at launch ('tf32x3' launches "
                                                  "there and is f32-class PER CELL; what an end-to-end model does with it is the kit's evidence)"},
                            # the triangle multiplication's 'ieee' kernels launch on this line (their HIGHEST dots are smaller): served, f32-class per cell
            "evidence": "a float32 dot-precision probe + core `pytest tests/` on the float32-model JAX image (jax 0.5.3 / jaxlib 0.5.3, H100 80GB HBM3)"},
    "0.10": {"tf32_class": "body", "ratio": 1.25, "ceiling": {}, "unlaunchable": {},          # round-to-nearest tf32 in Pallas-Triton: per-dot 2.9e-4 == XLA's
             "cause": "Pallas-Triton f32 dot at DEFAULT/tf32 rounds operands to nearest tf32: per-dot rel-rms 2.9e-4 == XLA's DEFAULT f32 GEMM",
             "evidence": "JAXF32 f32_final_02 (166 passed) + core055_jaxsuite_01 (jax 0.10.2, H100): fused tf32 2.3e-4 == the XLA-default body's"},
}
F32_LINE_DEFAULT_AT_OR_ABOVE = "0.10"                          # jax lines >= this without their own row read the "0.10" row (same lowering family); below it, only listed rows
N_NOT_MULTIPLE, C_NOT_MULTIPLE, C_OUT_OF_RANGE = "n_not_tile_multiple", "c_not_multiple_of_16", "c_out_of_range"
MASK_SHAPE, HEAD_DIM_UNSUPPORTED, HD_TOO_LARGE = "mask_shape", "head_dim_unsupported", "heads_x_dim_gt_256"
BACKEND_NOT_GPU, KERNEL_IMPORT_FAILED = "backend_not_gpu", "kernel_import_failed"
BIAS_KEYS_INCOMPLETE, UNKNOWN_KEY_MASK = "bias_keys_incomplete", "unknown_key_mask"
KEY_MASKS = ("by_column", "by_line")                 # key k of batch line b <- pair_mask[k, b] (by_column) | the attended line's own entry (by_line)


class Refusal(RuntimeError):
    """A named refusal (``kind``, ``detail``): the served_reason names, jax_missing · pallas_missing · backend_not_gpu · kernel_import_failed ·
    unknown_equation · unknown_orientation."""
    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{NAME}: {kind}" + (f" — {detail}" if detail else ""))


# ------------------------------------------------------------------------------------------------------------------ tables
F32_BY_JAX_LINE = "f32_by_jax_line"                                    # a table's optional per-jax-line f32 rows: {<jax line>: {attn_f32_default, attn_f32_by_n, trimul_f32, trimul_f32_by_n}}
_LINE_TABLES: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _line_table(cc: str) -> Dict[str, Any]:
    """``TILE_TABLES[cc]`` as served on this jax line: a table with ``f32_by_jax_line`` rows for ``jax_line()`` answers with those f32 rows in
    place of its base f32 rows (the same table object on every call); a table without them, or a line it names no rows for, is served as is."""
    t = TILE_TABLES[cc]
    by_line = t.get(F32_BY_JAX_LINE)
    if not by_line:
        return t
    line = jax_line()
    if line not in by_line:
        return t
    key = (cc, line)
    if key not in _LINE_TABLES:
        merged = {k: v for k, v in t.items() if k != F32_BY_JAX_LINE}
        merged.update(by_line[line])
        _LINE_TABLES[key] = merged
    return _LINE_TABLES[key]


def generation(cc: Optional[str]) -> Optional[str]:
    """The generation key of a compute capability — ``"8.6"`` -> ``"8.x"`` — when GENERATION_SAFE_TABLES names it; ``None`` for a capability
    below 8.0, unparsable or empty."""
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", cc or "")
    if not m:
        return None
    gen = f"{int(m.group(1))}.x"
    return gen if gen in GENERATION_SAFE_TABLES else None


def safe_row(cc: Optional[str], family: str, dtype: Any = SERVED_DTYPE) -> Dict[str, Any]:
    """The generation SAFE row for ``family`` (``"trimul"`` | ``"attn"`` / ``"triattn"``) at ``dtype`` (bfloat16 | float32) on compute capability
    ``cc`` — a copy; the same row whether or not ``cc`` has a tuned table of its own. Raises ``Refusal(no_tiles)`` for a capability of no named
    generation. (The accessor a shared safe-settings helper calls; tables_for serves these rows by itself when ``cc`` has no table.)"""
    gen = generation(cc)
    if gen is None:
        raise Refusal(NO_TILES, f"no-tiles:{cc or '?'} (no generation safe rows for this compute capability; generations: {sorted(GENERATION_SAFE_TABLES)})")
    base = {"trimul": "trimul", "attn": "attn", "triattn": "attn"}[family]
    row, _ = _rows(GENERATION_SAFE_TABLES[gen], base, dtype, cc)
    return row


_SAFE_NET = None                                                # the process's safe_settings.SafeNet for this lever (lazily: the helper imports nothing heavy)


def safe_net():
    """This lever's ``opt_core.kernels.safe_settings.SafeNet`` (one per process): ``.on`` / ``.word()`` (``"safe:no_cell:tile_table"`` once the
    generation rows serve — the census word ``settings=<word>`` of a kit's SERVED line) / ``.snapshot()``."""
    global _SAFE_NET
    if _SAFE_NET is None:
        from . import safe_settings as _helper  # noqa: WPS433
        _SAFE_NET = _helper.SafeNet(NAME, refused=lambda msg: Refusal(NO_TILES, msg))
    return _SAFE_NET


def _serve_safe(cc: str, gen: str) -> Dict[str, Any]:
    """The generation safe rows for this part THROUGH the shared helper (opt_core.kernels.safe_settings): ``SafeNet.no_cell`` prints its ONE
    stderr line the first time in the process — ``[opt_core/fpf_pallas] safe settings served (no_cell:tile_table, cc <cc>, triton ?)`` — carries
    the census word (``safe_net().word()`` = ``"safe:no_cell:tile_table"``) and hands back the rows; nothing here words or prints that line."""
    from . import safe_settings as _helper  # noqa: WPS433
    _SAFE_ANNOUNCED.setdefault(cc, gen)
    return safe_net().no_cell(SAFE_SHAPE_WORD, lambda rows: rows, GENERATION_SAFE_TABLES[gen], where=_helper.where_word(cc, None))


def tables_for(cc: Optional[str] = None) -> Tuple[str, Dict[str, Any], bool]:
    """``(cc, table, own)``: the tile table for compute capability ``cc`` (``"9.0"`` ...; ``None`` = the first jax device's).
    Resolution: (1) the capability's OWN table (``own=True``); (2) a capability without one but of a named generation (``generation``:
    8.x / 9.x / 10.x) is SERVED the generation's safe rows — ``own=False``, ``tiles_label`` ``"safe:<generation>"``, the shared helper's one
    ``safe settings served`` line (``_serve_safe``: opt_core.kernels.safe_settings SafeNet.no_cell) — the lever engaged, never refused; (3) any other capability is REFUSED by name — ``Refusal(kind="no_tiles",
    detail="no-tiles:<cc> ...")`` naming the two ways out (``OPT_CORE_PALLAS_ALLOW_FALLBACK_CC=1``: the ``"9.0"`` table with ``own=False``,
    recorded as ``"fallback:<cc>->9.0"``; or the kit's stock mode). Never a silent fallback."""
    import os  # noqa: WPS433
    cc = cc if cc is not None else compute_capability()
    if cc in TILE_TABLES:
        return cc, _line_table(cc), True
    gen = generation(cc)
    if gen is not None:
        return cc, _serve_safe(cc, gen), False
    if os.environ.get(FALLBACK_ENV, "").strip() == "1":
        return cc, TILE_TABLES[FALLBACK_CC], False
    raise Refusal(NO_TILES, f"no-tiles:{cc or '?'} (tile tables: {sorted(TILE_TABLES)}; generation safe rows: {sorted(GENERATION_SAFE_TABLES)}; "
                            f"set {FALLBACK_ENV}=1 to run the {FALLBACK_CC} table on this part, recorded as tiles={tiles_label(cc, own=False)}, "
                            f"or run the kit's stock mode (--mode off))")


def tiles_label(cc: Optional[str] = None, own: Optional[bool] = None) -> str:
    """The SERVED-line word for the tiles in use: ``"own:<cc>"``, ``"safe:<generation>"`` (the generation safe rows) or
    ``"fallback:<cc>->9.0"`` (the opt-in); raises ``no_tiles`` as ``tables_for``."""
    if own is None:
        cc, _, own = tables_for(cc)
    if own:
        return f"own:{cc}"
    gen = generation(cc)
    return f"safe:{gen}" if gen is not None else f"fallback:{cc or '?'}->{FALLBACK_CC}"


def compute_capability() -> str:
    """``str(jax.devices()[0].compute_capability)`` or ``""`` (no jax / no device)."""
    try:
        import jax  # noqa: WPS433 (lazy by contract)
        return str(getattr(jax.devices()[0], "compute_capability", "") or "")
    except Exception:  # noqa: BLE001
        return ""


def _rows(t: Dict[str, Any], base: str, dtype: Any, cc_: Optional[str]):
    """(default row, by-N rows) of table ``t`` for ``base`` (``"trimul"`` | ``"attn"``) at ``dtype``: the bf16 rows, or the ``<base>_f32`` rows for
    float32 (refused by name when the table has none)."""
    f32 = _dtype_name(dtype) == F32_DTYPE
    key = {"trimul": ("trimul_f32", "trimul_f32_by_n") if f32 else ("trimul", "trimul_by_n"),
           "attn": ("attn_f32_default", "attn_f32_by_n") if f32 else ("attn_default", "attn_by_n")}[base]
    if key[0] not in t:
        raise Refusal(NO_TILES, f"no-f32-tiles:{cc_ or '?'} ({base}; f32 rows: {sorted(c for c, tb in TILE_TABLES.items() if key[0] in tb)})")
    return dict(t[key[0]]), dict(t.get(key[1], {}))


def trimul_cfg(n: int, cc: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, dtype: Any = SERVED_DTYPE) -> Dict[str, Any]:
    """The triangle-multiplication tile config for pair size ``n``: table row, then its per-N row, then ``overrides``. ``dtype="float32"`` reads
    the f32 kernels' rows (``trimul_f32`` / ``trimul_f32_by_n``); a table without them is the refusal ``no_tiles`` (``no-f32-tiles:<cc>``)."""
    cc_, t, _ = tables_for(cc)
    out, by_n = _rows(t, "trimul", dtype, cc_)
    out.update({k: v for k, v in by_n.get(int(n), {}).items()})
    out.update(overrides or {})
    return out


def attn_cfg(n: int, cc: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None, dtype: Any = SERVED_DTYPE) -> Dict[str, Any]:
    """The triangle-attention tile config for pair size ``n``: table row, then its per-N row, then ``overrides``. ``dtype="float32"`` reads
    the f32 kernels' rows (``attn_f32_default`` / ``attn_f32_by_n``); a table without them is the refusal ``no_tiles`` (``no-f32-tiles:<cc>``)."""
    cc_, t, _ = tables_for(cc)
    out, by_n = _rows(t, "attn", dtype, cc_)
    out.update({k: v for k, v in by_n.get(int(n), {}).items()})
    out.update(overrides or {})
    return out


def required_multiple(kind: str, n: Optional[int] = None, cc: Optional[str] = None, dtype: Any = SERVED_DTYPE) -> int:
    """The pair-size multiple the tiles require for ``kind`` (``"trimul"``: lcm(t1, t2 [, tm, tn, tk with the Pallas GEMM]);
    ``"triattn"``: lcm(t1, t2, bq, bk) of the ``dtype`` route's rows) on compute capability ``cc`` at pair size ``n`` (a by-N row applies at
    that ``n`` only). 64 on the ``"9.0"`` table; raises ``no_tiles`` as ``tables_for`` / ``attn_cfg``."""
    import math  # noqa: WPS433
    if kind == "trimul":
        c = trimul_cfg(int(n) if n is not None else -1, cc, dtype=dtype)
        tiles = [c["t1"], c["t2"]] + ([c["tm"], c["tn"], c["tk"]] if c.get("ein") == "pallas" else [])
    else:
        c = attn_cfg(int(n) if n is not None else -1, cc, dtype=dtype)
        tiles = [c["t1"], c["t2"], c["bq"], c["bk"]]
    out = 1
    for t in tiles:
        out = out * int(t) // math.gcd(out, int(t))
    return out


def kernel_n(kind: str, n: int, cc: Optional[str] = None, dtype: Any = SERVED_DTYPE) -> int:
    """The pair size the kernels run at for a model pair size ``n``: ``n`` rounded UP to ``required_multiple(kind, n_kernel, cc, dtype)`` (the
    blocks zero-pad ``act`` and the mask to it and slice the result back — exact: padded keys/planes are masked, padded pixels sliced away).
    A kit's SERVED row records both ``n`` and ``kernel_n``."""
    m = required_multiple(kind, None, cc, dtype)
    nk = -(-int(n) // m) * m
    m2 = required_multiple(kind, nk, cc, dtype)                         # a by-N row at the padded size may ask a larger multiple
    return nk if nk % m2 == 0 else -(-int(n) // m2) * m2


# ------------------------------------------------------------------------------------------------------------------ served predicate
def _dtype_name(dtype: Any) -> str:
    return str(getattr(dtype, "name", dtype))


def head_dim(c: int, num_head: int) -> int:
    """The head dim the attention block serves for ``c`` channels over ``num_head`` heads: ``max(c // num_head, 16)``."""
    return max(int(c) // int(num_head), 16)


def served_reason(kind: str, act_shape: Sequence[int], act_dtype: Any, mask_shape: Optional[Sequence[int]],
                  num_head: Optional[int] = None, cc: Optional[str] = None, pad: bool = True) -> Optional[str]:
    """``None`` when a block of ``kind`` (``"trimul"`` | ``"triattn"``) serves this call, else the refusal's name. ``act_shape`` ``[N,N,C]``,
    ``mask_shape`` ``[N,N]`` (``None`` = no mask: refused, the kernels read one), ``num_head`` required for ``"triattn"``. ``cc``: the compute
    capability whose tile table decides the pair-size multiple (``required_multiple``); ``None`` = the ``"9.0"`` table's 64 (stdlib answer,
    no device query) — the blocks themselves pass the device's ``cc``. ``pad`` (the blocks' default): any ``N`` is served, the kernels run at
    ``kernel_n(kind, N, cc)``; ``pad=False``: ``N`` itself must be a tile multiple (``n_not_tile_multiple`` otherwise)."""
    shape = tuple(int(s) for s in act_shape)
    if len(shape) != 3:
        return RANK_NOT_3
    if _dtype_name(act_dtype) not in SERVED_DTYPES.get(kind, (SERVED_DTYPE,)):
        return DTYPE_NOT_SERVED
    n, n2, c = shape
    if n != n2:
        return NOT_SQUARE
    if not pad and n % (required_multiple(kind, n, cc, act_dtype) if cc is not None else N_MULTIPLE) != 0:
        return N_NOT_MULTIPLE
    if c % 16 != 0:
        return C_NOT_MULTIPLE
    if not (C_RANGE[0] <= c <= C_RANGE[1]):
        return C_OUT_OF_RANGE
    if mask_shape is None or tuple(int(s) for s in mask_shape) != (n, n):
        return MASK_SHAPE
    if kind == "triattn":
        if num_head is None:
            return HEAD_DIM_UNSUPPORTED
        d = head_dim(c, num_head)
        if d not in HEAD_DIMS:
            return HEAD_DIM_UNSUPPORTED
        if int(num_head) * d > MAX_HD:
            return HD_TOO_LARGE
    return None


# ------------------------------------------------------------------------------------------------------------------ kernel access (imports jax)
_MODS: Dict[str, Any] = {}


def _routed_names():
    from . import routed  # noqa: WPS433
    return routed()


def _load(file: str):
    """The carried module ``fpf_pallas.<file>``: the routed top-level package when a kit routed the name, else this package's copy."""
    if file in _MODS:
        return _MODS[file]
    import sys
    pkg_name = NAME if (NAME in sys.modules or NAME in _routed_names()) else __package__ + "." + NAME
    try:
        mod = importlib.import_module(pkg_name + "." + file)
    except ImportError as e:
        kind = "jax_missing" if getattr(e, "name", "") in ("jax", "jaxlib") else KERNEL_IMPORT_FAILED
        raise Refusal(kind, repr(e)[:300]) from e
    _MODS[file] = mod
    return mod


def kernels():
    """``(trimul_pallas, triattn_pallas)``: the two carried kernel modules (imports jax + jax.experimental.pallas.triton)."""
    return _load(TRIMUL_FILE), _load(TRIATTN_FILE)


def glut_module():
    """The carried ``pallas_glut`` module (imports jax AND tokamax)."""
    if GLUT_NAME in _MODS:
        return _MODS[GLUT_NAME]
    import sys
    name = GLUT_NAME if (GLUT_NAME in sys.modules or GLUT_NAME in _routed_names()) else __package__ + "." + GLUT_NAME
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        raise Refusal("tokamax_missing" if "tokamax" in repr(e) else KERNEL_IMPORT_FAILED, repr(e)[:300]) from e
    _MODS[GLUT_NAME] = mod
    return mod


def probe(require_gpu: bool = True) -> Dict[str, Any]:
    """``{ok, kind, detail, jax, backend, cc, own_table}`` — never raises."""
    out: Dict[str, Any] = {"ok": False, "kind": None, "detail": "", "jax": None, "backend": None, "cc": "", "own_table": False}
    try:
        import jax  # noqa: WPS433
        out["jax"] = getattr(jax, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        out.update(kind="jax_missing", detail=repr(e)[:200])
        return out
    try:
        from jax.experimental.pallas import triton as _plgpu  # noqa: F401, WPS433
    except Exception as e:  # noqa: BLE001
        out.update(kind="pallas_missing", detail=repr(e)[:200])
        return out
    try:
        out["backend"] = jax.default_backend()
    except Exception as e:  # noqa: BLE001
        out["backend"] = f"unknown:{e!r}"[:60]
    if require_gpu and out["backend"] != "gpu":
        out.update(kind=BACKEND_NOT_GPU, detail=f"jax.default_backend() = {out['backend']!r}; the Pallas ops lower on gpu only")
        return out
    try:
        kernels()
    except Refusal as r:
        out.update(kind=r.kind, detail=r.detail)
        return out
    try:
        cc, _, own = tables_for()
    except Refusal as r:                                               # no_tiles: this part has no table and no opt-in
        out.update(kind=r.kind, detail=r.detail, cc=compute_capability())
        return out
    out.update(ok=True, cc=cc, own_table=own, tiles=tiles_label(cc, own))
    return out


def require(require_gpu: bool = True) -> Dict[str, Any]:
    p = probe(require_gpu=require_gpu)
    if not p["ok"]:
        raise Refusal(p["kind"] or KERNEL_IMPORT_FAILED, p["detail"])
    return p


# ------------------------------------------------------------------------------------------------------------------ parameter builders
def attn_params_from_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """The carried converter: a bias-free-layout parameter tree -> the kernel dict (see module doc)."""
    return kernels()[1].attn_params_from_haiku(tree)


def attn_params(*, ln_scale, ln_offset, q_w, k_w, v_w, bias_w, gate_w, out_w, gate_b=None, out_b=None, weights_dtype=None) -> Dict[str, Any]:
    """Math-layout weights -> the kernel dict: ``q_w/k_w/v_w/gate_w [C,H,D]``, ``bias_w [C,H]``, ``out_w [H,D,C]``, LayerNorm ``[C]``; optional
    ``gate_b [H,D]`` + ``out_b [C]`` (both or neither) -> ``bg [HD]``, ``bo [C]`` f32 = the +bias epilogue. ``weights_dtype``: the dtype the
    weight matrices are stored in (``None`` = bfloat16, what the carried bf16 kernels read; ``jnp.float32`` for the f32 route)."""
    import jax.numpy as jnp  # noqa: WPS433
    f32 = jnp.float32
    bf16 = jnp.bfloat16 if weights_dtype is None else weights_dtype
    C, H, D = (int(s) for s in q_w.shape)
    HD = H * D
    wb16 = jnp.zeros((C, 16), bias_w.dtype).at[:, :H].set(bias_w)
    out = dict(ln_scale=ln_scale.astype(f32), ln_offset=ln_offset.astype(f32),
               wq_t=q_w.reshape(C, HD).astype(bf16), wk_t=k_w.reshape(C, HD).astype(bf16), wv2=v_w.reshape(C, HD).astype(bf16),
               wb16=wb16.astype(bf16), wg_t=gate_w.reshape(C, HD).astype(bf16), wo=out_w.reshape(HD, C).astype(bf16), H=H, D=D)
    if (gate_b is None) != (out_b is None):
        raise Refusal(BIAS_KEYS_INCOMPLETE, "attn_params: gate_b and out_b go together")
    if gate_b is not None:
        out.update(bg=gate_b.reshape(HD).astype(f32), bo=out_b.reshape(C).astype(f32))
    return out


def trimul_params(*, ln_in_scale, ln_in_offset, left_w, right_w, left_gate_w, right_gate_w, ln_c_scale, ln_c_offset, out_w, gate_w,
                  left_b=None, right_b=None, left_gate_b=None, right_gate_b=None, out_b=None, gate_b=None) -> Dict[str, Any]:
    """Separate ``[C,C]`` left/right projection and gate matrices -> the interleaved ``w_proj/w_gate [C,2C]`` dict the trimul block reads
    (column ``2c`` = left channel ``c``, column ``2c+1`` = right channel ``c``); the six optional bias vectors ``[C]`` (all or none) ->
    ``b_proj/b_gate [2C]`` (interleaved the same way), ``b_out``, ``b_gl`` = the +bias variant."""
    import jax.numpy as jnp  # noqa: WPS433
    C = int(left_w.shape[0])

    def interleave(a, b):
        return jnp.stack([a, b], axis=-1).reshape(a.shape[:-1] + (2 * C,))
    out = dict(ln_in_scale=ln_in_scale, ln_in_offset=ln_in_offset, w_proj=interleave(left_w, right_w), w_gate=interleave(left_gate_w, right_gate_w),
               ln_c_scale=ln_c_scale, ln_c_offset=ln_c_offset, w_out=out_w, w_gl=gate_w)
    biases = [left_b, right_b, left_gate_b, right_gate_b, out_b, gate_b]
    if any(b is not None for b in biases):
        if any(b is None for b in biases):
            raise Refusal(BIAS_KEYS_INCOMPLETE, "trimul_params: the six biases go together")
        out.update(b_proj=interleave(left_b, right_b), b_gate=interleave(left_gate_b, right_gate_b), b_out=out_b, b_gl=gate_b)
    return out


# ------------------------------------------------------------------------------------------------------------------ blocks
def _refuse_unserved(kind: str, act, mask, num_head: Optional[int] = None, cc: Optional[str] = None, pad: bool = True) -> str:
    cc = tables_for(cc)[0]                                              # the device's table (or no_tiles) decides the pair-size multiple
    r = served_reason(kind, act.shape, act.dtype, None if mask is None else mask.shape, num_head=num_head, cc=cc, pad=pad)
    if r is not None:
        raise Refusal(r, f"{kind}: act {tuple(act.shape)} {_dtype_name(act.dtype)}, mask {None if mask is None else tuple(mask.shape)}, "
                         f"pair-size multiple {required_multiple(kind, int(act.shape[0]) if len(act.shape) == 3 else None, cc, act.dtype)} on cc {cc}")
    return cc


def _padded(act, mask, nk: int):
    """``act`` / ``mask`` zero-padded from ``[N,N,·]`` to ``[nk,nk,·]`` (a no-op at ``nk == N``)."""
    import jax.numpy as jnp  # noqa: WPS433
    n = int(act.shape[0])
    if nk == n:
        return act, mask
    p = nk - n
    return jnp.pad(act, ((0, p), (0, p), (0, 0))), jnp.pad(mask, ((0, p), (0, p)))


def trimul_block(act, mask, params: Dict[str, Any], *, equation: str, cfg: Optional[Dict[str, Any]] = None, cc: Optional[str] = None,
                 precision: str = "std", pad: bool = True):
    """Fused triangle multiplication (module doc). ``cfg`` overrides the table row for this call; ``precision="hi"`` = f32 hand-off of the
    contraction result (bf16 route) or the MMA operand precision word (f32 route, ``f32_precision``); ``pad``: run the kernels at
    ``kernel_n('trimul', N, cc, dtype)`` (zero-pad + slice back, exact) when ``N`` is not a tile multiple. float32 ``act`` runs the f32
    kernels (``fpf_pallas_f32``), bfloat16 ``act`` the carried ones. ``Refusal`` (named) outside the served shapes/dtype."""
    if equation not in EQUATIONS:
        raise Refusal("unknown_equation", repr(equation))
    cc = _refuse_unserved("trimul", act, mask, cc=cc, pad=pad)
    from . import fpf_pallas_bias as B  # noqa: WPS433
    hb = B.has_bias(params, "trimul")
    if hb is None:
        raise Refusal(BIAS_KEYS_INCOMPLETE, f"trimul biases present must be all of {B.TRIMUL_BIAS_KEYS}")
    f32_route = _dtype_name(act.dtype) == F32_DTYPE
    mma = f32_precision(precision, "trimul") if f32_route else None
    n = int(act.shape[0]); nk = kernel_n("trimul", n, cc, act.dtype) if pad else n
    a, m = _padded(act, mask, nk)
    c = trimul_cfg(nk, cc, cfg, dtype=act.dtype)
    if f32_route:                                                       # the f32 kernels (biases, present or not, ride the same kernel body)
        from . import fpf_pallas_f32 as F  # noqa: WPS433
        out = F.triangle_multiplication_fused_f32(a, m, params, equation=EQUATIONS[equation], cfg=c, precision=mma)
        return (out if nk == n else out[:n, :n]).astype(act.dtype)
    c["f32_tri"] = precision == "hi"
    if hb:                                                              # the +bias variant (kernels/fpf_pallas_bias.py)
        out = B.triangle_multiplication_fused_bias(a, m, params, equation=EQUATIONS[equation], cfg=c)
    else:
        out = kernels()[0].triangle_multiplication_fused(a, m, params, equation=EQUATIONS[equation], cfg=c)
    return (out if nk == n else out[:n, :n]).astype(act.dtype)


def _kernel_pair_mask(pair_mask, orientation: str, key_mask: str):
    """The mask argument the kernels read (their rule: key ``k`` of batch line ``b`` is valid iff ``arg[k, b] > 0``) for the caller's
    convention: ``by_column`` = ``pair_mask[k, b]`` for both orientations (the stock line ``mask = swapaxes(pair_mask)[b]``);
    ``by_line`` = the attended line's own entry (``pair_mask[b, k]`` for the starting node's rows, ``pair_mask[k, b]`` for the ending node's
    columns). Identical for symmetric masks."""
    if key_mask not in KEY_MASKS:
        raise Refusal(UNKNOWN_KEY_MASK, repr(key_mask))
    if key_mask == "by_line" and not ORIENTATIONS[orientation]:
        import jax.numpy as jnp  # noqa: WPS433
        return jnp.swapaxes(pair_mask, -1, -2)
    return pair_mask


def _f32_module():
    """The f32 kernels module (standard library at import: reading its vocabulary imports no jax)."""
    return importlib.import_module(__name__.rsplit(".", 1)[0] + ".fpf_pallas_f32")


def __getattr__(name):                                          # PEP 562: F32_PRECISIONS is the kernel module's tuple, never re-typed here
    if name == "F32_PRECISIONS":
        return _f32_module().PRECISIONS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def f32_precision(precision: Optional[str], kind: Optional[str] = None) -> str:
    """The f32 route's MMA operand precision word for ``precision``: REQUIRED — ``"std"`` / ``None`` is the refusal ``precision_required`` (the
    bf16 route's default means nothing here; every f32 word is a deliberate speed / accuracy class: ``fpf_pallas_f32.PRECISION_WORDS_HELP``);
    ``F32_PRECISIONS`` pass; anything else is the refusal ``unknown_precision``; with ``kind``, a word whose kernel cannot launch on this jax
    line (``F32_LINES[line]["unlaunchable"]``) is the refusal ``precision_unlaunchable``."""
    F = _f32_module()
    if precision in ("std", None):
        raise Refusal(PRECISION_REQUIRED, "f32 route: name precision= one of " + F.PRECISION_WORDS_HELP)
    if precision not in F.PRECISIONS:
        raise Refusal(UNKNOWN_PRECISION, f"f32 route: precision {precision!r} not in {F.PRECISIONS} — " + F.PRECISION_WORDS_HELP)
    if kind is not None:
        row = f32_line_row()
        why = (row or {}).get("unlaunchable", {}).get((kind, precision))
        if why:
            raise Refusal(PRECISION_UNLAUNCHABLE, f"f32 route: precision {precision!r} for {kind} on jax {jax_line()}: {why}")
    return precision


def jax_line(version: Optional[str] = None) -> str:
    """The jax release line 'major.minor' of ``version`` (default: the imported / importable jax's ``__version__``; 'none' without jax)."""
    v = version
    if v is None:
        j = sys.modules.get("jax")
        if j is None:
            try:
                j = importlib.import_module("jax")
            except Exception:  # noqa: BLE001
                return "none"
        v = getattr(j, "__version__", "") or ""
    parts = re.match(r"^(\d+)\.(\d+)", str(v))
    return f"{int(parts.group(1))}.{int(parts.group(2))}" if parts else "none"


def f32_line_row(version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The F32_LINES row for this jax line: its own row, else the default row when the line is at or above F32_LINE_DEFAULT_AT_OR_ABOVE, else None."""
    line = jax_line(version)
    if line in F32_LINES:
        return F32_LINES[line]
    def _t(s):
        return tuple(int(x) for x in s.split("."))
    try:
        if _t(line) >= _t(F32_LINE_DEFAULT_AT_OR_ABOVE):
            return F32_LINES[F32_LINE_DEFAULT_AT_OR_ABOVE]
    except ValueError:
        pass
    return None


def f32_class(precision: str, version: Optional[str] = None) -> str:
    """The numerics-class word of the f32 route at ``precision`` on this jax line — the word a kit prints beside ``mma=<precision>`` on its
    SERVED line (``f32_class=<word>``): 'body' | 'bf16op' (at 'tf32', per F32_LINES) | 'f32' ('tf32x3' / 'ieee') | 'unlisted:<line>'."""
    if precision in ("tf32x3", "ieee"):
        return "f32"
    row = f32_line_row(version)
    if row is None:
        return "unlisted:" + jax_line(version)
    return str(row["tf32_class"])


def tri_attn_block(act, pair_mask, params: Dict[str, Any], *, orientation: str, ending_bias_transposed: bool = True, key_mask: str = "by_column",
                   cfg: Optional[Dict[str, Any]] = None, cc: Optional[str] = None, precision: str = "std", pad: bool = True):
    """Fused triangle self-attention (module doc). ``params`` from ``attn_params`` / ``attn_params_from_tree``; gating / output biases in
    ``params`` (``bg``, ``bo``) select the +bias epilogue; ``pad`` as ``trimul_block`` (padded keys are masked, padded rows sliced away).
    float32 ``act`` runs the f32 kernels (``fpf_pallas_f32``; ``precision`` = the MMA operand precision word, ``f32_precision``; biases,
    present or not, ride the same kernel body); bfloat16 ``act`` the carried kernels."""
    if orientation not in ORIENTATIONS:
        raise Refusal("unknown_orientation", repr(orientation))
    cc = _refuse_unserved("triattn", act, pair_mask, num_head=params["H"], cc=cc, pad=pad)
    from . import fpf_pallas_bias as B  # noqa: WPS433
    hb = B.has_bias(params, "triattn")
    if hb is None:
        raise Refusal(BIAS_KEYS_INCOMPLETE, f"attention biases present must be all of {B.ATTN_BIAS_KEYS}")
    f32_route = _dtype_name(act.dtype) == F32_DTYPE
    mma = f32_precision(precision, "triattn") if f32_route else None
    n = int(act.shape[0]); nk = kernel_n("triattn", n, cc, act.dtype) if pad else n
    a, pm = _padded(act, _kernel_pair_mask(pair_mask, orientation, key_mask), nk)
    c = attn_cfg(nk, cc, cfg, dtype=act.dtype)
    if f32_route:
        from . import fpf_pallas_f32 as F  # noqa: WPS433
        out = F.grid_self_attention_fused_f32(a, pm, params, transpose=ORIENTATIONS[orientation], ending_bias_transposed=bool(ending_bias_transposed), cfg=c, precision=mma)
        return (out if nk == n else out[:n, :n]).astype(act.dtype)
    c["f32_o"] = precision == "hi"
    if hb:
        out = B.grid_self_attention_fused_bias(a, pm, params, transpose=ORIENTATIONS[orientation], ending_bias_transposed=bool(ending_bias_transposed), cfg=c)
    else:
        out = kernels()[1].grid_self_attention_fused(a, pm, params, transpose=ORIENTATIONS[orientation], ending_bias_transposed=bool(ending_bias_transposed), cfg=c)
    return (out if nk == n else out[:n, :n]).astype(act.dtype)


def attention_core(q, k, v, bias, key_mask, *, cfg: Optional[Dict[str, Any]] = None, cc: Optional[str] = None, out_dtype=None):
    """The flash core alone: ``q/k/v [B,S,H,D]``, ``bias [H,S,S]`` or ``[1,H,S,S]``, ``key_mask [B,S]`` bool -> ``[B,S,H,D]``."""
    A = kernels()[1]
    S = int(q.shape[1])
    c = attn_cfg(S, cc, cfg)
    b = bias if bias.ndim == 4 else bias[None]
    return A.flash_attention_bshd(q, k, v, b, key_mask, bq=c["bq"], bk=c["bk"], num_warps=c["wa"], num_stages=c["sa"], out_dtype=out_dtype)


def glu_transposed_masked(x, weights, mask2d, *, activation, config=None):
    """The carried ``pallas_glut`` kernel (module doc); imports tokamax."""
    return glut_module().glu_transposed_masked(x, weights, mask2d, activation=activation, config=config)


# ------------------------------------------------------------------------------------------------------------------ references (pure jnp)
def trimul_reference(act, mask, params: Dict[str, Any], *, equation: str, compute_dtype=None, precision=None):
    """Pure-jnp replica of the block ``trimul_block`` computes (biases included when ``params`` carries them): input LayerNorm (f32
    statistics, eps 1e-5) -> left/right projections (+bias) * mask * sigmoid(gates (+bias)) -> the triangle contraction -> centre LayerNorm ->
    output projection (+bias) * sigmoid(gating (+bias)). ``compute_dtype`` = the dtype every op's result is rounded to (``bfloat16`` mimics a
    stock bf16 module op by op; ``None``/f32 with ``precision=jax.lax.Precision.HIGHEST`` is the reference). Weights are rounded
    to ``act.dtype`` first (that is the model)."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    f32 = jnp.float32
    cd = compute_dtype if compute_dtype is not None else f32
    hi = dict(precision=precision, preferred_element_type=f32)
    r = lambda x: x.astype(cd).astype(f32)  # noqa: E731  (round to the compute dtype, keep computing in f32)
    rw = lambda w: w.astype(act.dtype).astype(f32)  # noqa: E731
    C = int(act.shape[-1])
    zeros_c, zeros_2c = jnp.zeros((C,), f32), jnp.zeros((2 * C,), f32)
    b_proj = params.get("b_proj", zeros_2c).astype(f32); b_gate = params.get("b_gate", zeros_2c).astype(f32)
    b_out = params.get("b_out", zeros_c).astype(f32); b_gl = params.get("b_gl", zeros_c).astype(f32)

    def ln(x, scale, offset):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True) - jnp.square(mean)
        return (x - mean) * jax.lax.rsqrt(var + 1e-5) * scale.astype(f32) + offset.astype(f32)
    x = r(ln(act.astype(f32), params["ln_in_scale"], params["ln_in_offset"]))
    proj = r(jnp.einsum("ijc,cd->ijd", x, rw(params["w_proj"]), **hi) + b_proj)
    gate = r(jax.nn.sigmoid(r(jnp.einsum("ijc,cd->ijd", x, rw(params["w_gate"]), **hi) + b_gate)))
    prj = r(r(proj * mask.astype(f32)[:, :, None]) * gate)
    left, right = prj[:, :, 0::2], prj[:, :, 1::2]
    tri = r(jnp.einsum(EQUATIONS[equation], r(left), r(right), **hi))
    y = r(ln(tri, params["ln_c_scale"], params["ln_c_offset"]))
    out = r(jnp.einsum("ijc,cd->ijd", y, rw(params["w_out"]), **hi) + b_out)
    g = r(jax.nn.sigmoid(r(jnp.einsum("ijc,cd->ijd", x, rw(params["w_gl"]), **hi) + b_gl)))
    return r(out * g).astype(act.dtype)


def tri_attn_reference(act, pair_mask, *, ln_scale, ln_offset, q_w, k_w, v_w, bias_w, gate_w, out_w, orientation: str,
                       ending_bias_transposed: bool = True, key_mask: str = "by_column", gate_b=None, out_b=None, precision=None,
                       row_chunk: Optional[int] = None):
    """Pure-jnp f32 replica of the block ``tri_attn_block`` computes, weights in the math layout of ``attn_params`` (rounded to ``act.dtype``
    first — that IS the model — then upcast): LayerNorm (f32 statistics, eps 1e-5) -> pair bias ``[H,q,k]`` from the UN-transposed normalised
    activation (transposed per head for the ending node when ``ending_bias_transposed``) -> rows of the (transposed, for the ending node)
    normalised activation attend over their positions with the key mask of the ``key_mask`` convention (``by_column``: ``pair_mask[k, b]``,
    the stock line; ``by_line``: the attended line's own entry; identical for symmetric masks), softmax scale ``1/sqrt(D)`` -> sigmoid gate from the same rows -> output projection -> transposed back.
    ``precision=jax.lax.Precision.HIGHEST`` for the reference. ``row_chunk``: attend ``row_chunk`` batch lines at a time
    (``jax.lax.map``; same math per line, bounds the ``[B,H,S,S]`` logits in memory at large ``N``)."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    f32 = jnp.float32
    T = ORIENTATIONS[orientation]
    C, H, D = (int(s) for s in q_w.shape)
    hi = dict(precision=precision, preferred_element_type=f32)
    rw = lambda w: w.astype(act.dtype).astype(f32)  # noqa: E731
    x = act.astype(f32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True) - jnp.square(mean)
    xln = (x - mean) * jax.lax.rsqrt(var + 1e-5) * ln_scale.astype(f32) + ln_offset.astype(f32)
    bias = jnp.einsum("qkc,ch->hqk", xln, rw(bias_w), **hi)
    if T and ending_bias_transposed:
        bias = jnp.swapaxes(bias, -1, -2)
    a = jnp.swapaxes(xln, 0, 1) if T else xln
    kmask = jnp.swapaxes(_kernel_pair_mask(pair_mask, orientation, key_mask), 0, 1) > 0     # [b, k]
    q = jnp.einsum("bsc,chd->bshd", a, rw(q_w), **hi) * (1.0 / (D ** 0.5))
    k = jnp.einsum("bsc,chd->bshd", a, rw(k_w), **hi)
    v = jnp.einsum("bsc,chd->bshd", a, rw(v_w), **hi)
    def attend(qc, kc, vc, kmc):                                        # a chunk of batch lines
        logits = jnp.einsum("bqhd,bkhd->bhqk", qc, kc, **hi) + bias[None]
        logits = jnp.where(kmc[:, None, None, :], logits, -1e9)
        return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), vc, **hi)
    B = int(q.shape[0])
    if row_chunk and B % int(row_chunk) == 0 and B // int(row_chunk) > 1:
        nb = B // int(row_chunk)
        rs = lambda x: x.reshape((nb, B // nb) + x.shape[1:])  # noqa: E731
        o = jax.lax.map(lambda t: attend(*t), (rs(q), rs(k), rs(v), rs(kmask))).reshape(q.shape)
    else:
        o = attend(q, k, v, kmask)
    g = jnp.einsum("bsc,chd->bshd", a, rw(gate_w), **hi) + (0.0 if gate_b is None else gate_b.astype(f32)[None, None])
    o = o * jax.nn.sigmoid(g)
    out = jnp.einsum("bshd,hdc->bsc", o, rw(out_w), **hi) + (0.0 if out_b is None else out_b.astype(f32)[None, None])
    if T:
        out = jnp.swapaxes(out, 0, 1)
    return out.astype(act.dtype)
