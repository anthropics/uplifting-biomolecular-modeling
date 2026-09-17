"""fpf_trimul.trimul — FPF_SPEC_v0 registry adapter for `trimul_out` / `trimul_in` (the stock class:
`TriangleMultiplicativeUpdate.forward(z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative='torch')`).

engine_stock_class reproduced: "<engine> cuEq-fused TriMul {outgoing|incoming} (the served path)" =
    z_in = z.clone() ; y = cuEq.triangle_multiplicative_update(z[None], direction, mask=ones|mask, LN_in, cat[a_p,b_p], cat[a_g,b_g], LN_out, linear_z.W, linear_g.W, eps=1e-5)[0]
    return y + z_in   (when inplace_safe and _add_with_inplace)  else y
Precision policy (the same observable numerics as the stock op): under autocast the weights AND the LN output are cast to the autocast dtype before the GEMMs (bf16 MMA);
without autocast fp32 weights run TF32 (RN-converted) if torch.backends.cuda.matmul.allow_tf32 else tf32x3.  Only `triangle_multiplicative='cuequivariance'`
with c_z == c_hidden is replaced; every other call goes to the stock method (fpf.original) unchanged.
Fixed tile table: CONFIG_TABLE below, keyed (dtype class, N-bucket); sha256 printed by config_table_sha().
"""
import hashlib, json, os, sys, torch
from . import kernels as K

# pinned table — pure function of (dtype, N bucket); no runtime autotune
V6_CFG = dict(A=dict(v=6, BM=128, BN=64, BK=64, num_warps=4, num_stages=3), B=dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=3, DMAJOR=True),
              C=dict(v=6, BM=64, BN=128, BK=64, num_warps=4, num_stages=3), LN=dict(BM=64, num_warps=4), LNO=dict(BM=64, num_warps=4),
              pad=int(os.environ.get("FPF_TRIMUL_PAD", "8")), contract=os.environ.get("FPF_TRIMUL_CONTRACT", "cublas"))
# ---- TILE TABLE: pinned per (dtype class, channel width C); mode overlays (exact|fast) on top.
# bf16 C=256 class: v9 = v6 arithmetic with a runtime (pipelined) K loop — A' 128x64x64 w4 s3 255-261 TF/s, C' 128x128x64 w8 s3 254-262 TF/s @705/813, bit-exact to v6.
# bf16 C=128 class: same tiles (swept at N = 268/554/705; v9 at C=128 not separately swept).
# fp32 C=384 class (TF32 MMA with RN-converted operands): fp32 operand tiles are 2x the bytes -> the bf16 tiles DO NOT FIT (OutOfResources: 294,912 B > 232,448 B).
#   Shared-memory budget per pipelined stage = (x tiles + weight tiles) * 4 B: A' 64x64x32 = (64*32 + 2*32*64)*4 = 24,576 B/stage -> 73,728 B at 3 stages; C' 64x64x32 (two x operands) = 32,768 B/stage -> 98,304 B.
#   LN passes at C=384 use CK = 512 (next pow2): BM=16 keeps the [16, 512] fp32 tile at 64 registers/thread (BM=64 spilled -> the 42 ms v1 reading).
TILES = {
    # bf16 C=256 class, v9 tiles (real dumps): A' 128x64x64 w4 s3, C' 128x128x64 w8 s3.
    # Measured (real dumps, both modes, 4 sizes x 2 dumps x 2 directions, stock brackets): A' v14 128x128x32 w4 s4 (interleaved single-acc) = 0.84/0.89 ms @705
    #   FAST/EXACT planes vs v9 1.04/1.05; bit-exact to _k_proj9 525/525; full op (bench_op) FAST 2.39 vs 2.65 ms @705, EXACT 3.90 vs 3.95. Pinned for C=256, both modes.
    ("bf16", 256): dict(A=dict(v=14, BM=128, BN=128, BK=32, num_warps=4, num_stages=4), C=dict(v=9, BM=128, BN=128, BK=64, num_warps=8, num_stages=3), LN=dict(BM=64, num_warps=4), LNO=dict(BM=64, num_warps=4)),
    # Not pinned: v9 A' 64x64x32 w4 s4 + C' 128x128x32 w8 s4 are 4.6 % faster than the v9 tiles above at 546 (FAST, pad 16) but 9 % slower than the v14 A' at 546.
    # bf16 C=128 class (synthetic z of engine shape, cuEq 0.8.0 functional stock in-process): best EXACT A' 64x64x32 w4 s3 + C' 64x64x32 w4 s3 = 1.867 ms @705 (x1.01),
    #   best FAST-cuBLAS A' 64x64x32 w4 s3 + C' 64x128x32 w4 s2 = 1.08 ms (x1.75); FAST-triton (pad 64: +25 % contraction FLOPs at 705->768) 1.34 ms (x1.41). In-engine (cuEq 0.10.0): EXACT bitwise 32/32.
    ("bf16", 128): dict(A=dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=3), C=dict(v=9, BM=64, BN=128, BK=32, num_warps=4, num_stages=2), LN=dict(BM=64, num_warps=4), LNO=dict(BM=64, num_warps=4)),
    # fp32 C=384 class (TF32 MMA with RNA-rounded operands = the cuBLAS class). ROW = the fp32-class sweep (REAL fp32 dumps
    #   b0|c0 + b47|c9, synthetic weights, stock = the engine's TUNED cuEq cache in-process): A'/C' 128x128x32 w8 s3 (144 / 192 KB smem — the largest fitting fp32 tiles;
    #   every BK>=64 tile with BM|BN>=128 at >=3 stages is OutOfResources in fp32).  @705 cuEq-only: stock-tuned 10.64 ms; EXACT 10.91 (x0.975, bit-exact) -> 'no candidate';
    #   FAST pad-8 cuBLAS 8.72 (x1.22, ratio 1.000/1.000; module-level x1.34 with the residual fused); @436 FAST x1.00-1.02 (parity cuEq-only, x1.11 module).
    #   64x64x32 w4 s3 tiles are parity vs the DEFAULT-cache stock but only 0.85x (EXACT) vs the tuned stock -> not pinned.
    ("fp32", 384): dict(A=dict(v=9, BM=128, BN=128, BK=32, num_warps=8, num_stages=3), C=dict(v=9, BM=128, BN=128, BK=32, num_warps=8, num_stages=3), B=dict(BM=128, BN=128, BK=32, num_warps=8, num_stages=3, DMAJOR=True), LN=dict(BM=16, num_warps=4), LNO=dict(BM=8, num_warps=4)),
}
TILES_DEFAULT = {"bf16": TILES[("bf16", 256)], "fp32": TILES[("fp32", 384)]}
# ---- per-architecture tile tables (arch_tables.json, keyed by compute capability: an entry may carry A'/C' tiles per "<cls>_<C>" cell — sm_100/sm_103: fp32_384; sm_80: fp32_384,
#      bf16_256, bf16_128, all <= the 163 KB sm_80 shared-memory budget); FPF_TRIMUL_ARCH_TABLE=0 disables (= shipped H100 table everywhere)
ARCH_TABLE_USED = None
try:
    _AT = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_tables.json")))
    if os.environ.get("FPF_TRIMUL_ARCH_TABLE", "1") != "0" and torch.cuda.is_available():
        _mj, _mn = torch.cuda.get_device_capability(0); _ak = "sm_%d%d" % (_mj, _mn)
        for (_cls, _C), _shipped in list(TILES.items()):                 # every "<cls>_<C>" key the arch entry carries (fp32_384; sm_80 also bf16_256 / bf16_128); null / absent = shipped tiles
            _ov = (_AT.get("tables", {}).get(_ak) or {}).get("%s_%d" % (_cls, _C))
            if _ov:
                TILES[(_cls, _C)] = dict(_shipped, A=dict(_ov["A"]), C=dict(_ov["C"]))
                if TILES_DEFAULT.get(_cls) is _shipped: TILES_DEFAULT[_cls] = TILES[(_cls, _C)]
                ARCH_TABLE_USED = _ak
except Exception as _e:   # never fail import on a table problem: shipped tiles, said ONCE (a table that cannot be read is named, not silent)
    ARCH_TABLE_USED = "error:" + repr(_e)[:80]
    print("[fpf_trimul] arch_tables.json unreadable (%s): the shipped H100 tiles serve every capability (arch_table=%s)" % (repr(_e)[:80], ARCH_TABLE_USED), file=sys.stderr, flush=True)
# Cell decision (ONE statement, cell_verdict): a VERIFIED (dtype class, C) cell — rows + numerics test record exist — is served; a MEASURED_OFF cell — a class
# certified engines run with this lever OFF — keeps the stock path BY NAME ('cell:<cls>_C<C>+off(not-measured)'); any OTHER bf16 / fp32 class is UNKNOWN and
# is SERVED on the default tiles, named 'unverified(<cls>_C<C>)' (opt_core.kernels.cell_words); a dtype outside bf16 / fp32 is outside what the kernels compute.
VERIFIED = {("bf16", 256): "bf16 C=256 class: T9 rows of record (real dumps) + partner blind certificate", ("bf16", 128): "bf16 C=128 class: Integration J6c (EXACT bitwise 32/32 in-engine) + cclass tiles", ("fp32", 384): "fp32 C=384 class: a tile sweep on real dumps vs the tuned kit stock (EXACT bitwise 16/16 but x0.98 = no candidate; FAST x1.22 @705 TIER2)"}
# EXACT-only cells: checked bit-exact vs the stock op on every call of the engine's equality panel (the kit's per-call check form), never
# served in FAST (no tier-2 row for them).
VERIFIED_EXACT = {("fp32", 64): "fp32 C=64 template-module TriMul: L3L4 per-call proof — FPF_VERIFY on every call of the identity panel "
                                "b200/b400/b600 under --det 1: 3 x 4992 calls per op, all bitwise, max|d| 0 (three boxes; "
                                "record archived)"}


MEASURED_OFF = {("bf16", 64): "template-stack TriMul under bf16 autocast (c=64): every certified engine composition runs the stock TriMul there",
                ("fp32", 64): "fp32 c=64 template-module TriMul outside EXACT mode: certified only through the exact per-call proof (VERIFIED_EXACT); FAST rows never measured",
                ("fp32", 128): "fp32 c_z=128 TriMul (fp32 trunks): the certified template-track compositions run the stock TriMul; no tile row measured",
                ("fp32", 256): "fp32 c_z=256 TriMul: the certified compositions run the stock TriMul; no tile row measured"}
STATS = {"calls": 0, "fallback": 0, "fallback_by": {}, "unverified": {}}     # the registry adapter's census: served calls, stock-path calls by reason word, UNKNOWN classes served by name


def dtype_class(dtype):
    """'bf16' | 'fp32' | the dtype's short name (a class the kernels do not compute)."""
    return "bf16" if dtype == torch.bfloat16 else ("fp32" if dtype == torch.float32 else str(dtype).replace("torch.", ""))


def cell_verdict(dtype, C):
    """The ONE cell decision for a (compute dtype, channel width C) class in this process's mode -> (kind, word):
    ("verified", <record>) a VERIFIED cell (VERIFIED_EXACT too in exact mode): served; ("off", "cell:<cls>_C<C>+off(not-measured)") a MEASURED_OFF class:
    the stock path BY NAME; ("unverified", "unverified(<cls>_C<C>)") any other bf16 / fp32 class: SERVED on the default tiles (TILES_DEFAULT via _select_cfg),
    named; ("unsupported", "dtype:<cls>") a dtype the kernels do not compute."""
    from opt_core.kernels.cell_words import with_off, unverified_word
    cls, C = dtype_class(dtype), int(C)
    rec = VERIFIED.get((cls, C)) or (VERIFIED_EXACT.get((cls, C)) if _MODE == "exact" else None)
    if rec is not None:
        return "verified", rec
    if cls not in ("bf16", "fp32"):
        return "unsupported", "dtype:%s" % cls
    if C % 64:                                                          # the A'/C' tiles step the channel axis in BK = 64 (bf16) / 32 (fp32) columns and the LN kernels in whole warps
        return "unsupported", "c:%d_not_multiple_of_64" % C
    if (cls, C) in MEASURED_OFF:
        return "off", with_off("cell:%s_C%d" % (cls, C))
    return "unverified", unverified_word("%s_C%d" % (cls, C))


def is_verified(dtype, C):
    """True for a VERIFIED cell of this process's mode (the certification record exists) — cell_verdict's first case."""
    return cell_verdict(dtype, C)[0] == "verified"


def _count_stock(word):
    STATS["fallback"] += 1
    STATS["fallback_by"][word] = STATS["fallback_by"].get(word, 0) + 1


def _note_unverified(word):
    """Count an UNKNOWN class served on the default tiles; ONE stderr line the first time each class is met in the process."""
    n = STATS["unverified"][word] = STATS["unverified"].get(word, 0) + 1
    if n == 1:
        print("[fpf_trimul] cells=%s: no certification record for this (dtype, C) class — served on the default tiles (%s mode), named, not refused"
              % (word, _MODE), file=sys.stderr, flush=True)


def verified_cells():
    """The served cells of this process's mode, for messages: 'bf16 C=256/128, fp32 C=384/64'."""
    cells = list(VERIFIED) + (list(VERIFIED_EXACT) if _MODE == "exact" else [])
    return ", ".join("%s C=%s" % (cls, "/".join(str(c) for cl, c in cells if cl == cls)) for cls in dict.fromkeys(cl for cl, _c in cells))
MODES = {"exact": dict(pad=1, ln_mode="stock", contract="cublas"),                      # bit-exact to the stock module (stock LN kernels + our A'/C' + unpadded cuBLAS)
         "fast": dict(pad=int(os.environ.get("FPF_TRIMUL_PAD", "16")), contract=os.environ.get("FPF_TRIMUL_CONTRACT", "cublas"))}   # TIER2_SAME_CLASS. FAST default = cuBLAS on pad-16 planes (fastest class once A' is despecialised; pad 64 cuBLAS is +13 % @813 in). FPF_TRIMUL_CONTRACT=triton (pad 64, Triton K_B) = the pad-INVARIANT arm (blind-tested pad-invariance record) for ragged batching.
V9_CFG = dict(V6_CFG, **TILES[("bf16", 256)], **MODES["fast"])
EXACT_CFG = dict(V6_CFG, **TILES[("bf16", 256)], **MODES["exact"])
_MODE = os.environ.get("FPF_TRIMUL_MODE", "exact")    # DEFAULT exact | fast = opt-in
assert _MODE in MODES, _MODE
_DEF = EXACT_CFG if _MODE == "exact" else V9_CFG
CONTRACT = os.environ.get("FPF_TRIMUL_CONTRACT", "cublas")   # EXACT: unpadded cuBLAS (bit-exact construction); FAST: cuBLAS on pad-16 planes (default) | 'triton' = Triton K_B on pad-64 planes (pad-invariant arm)
def _build_config_table(mode):
    tab = {"bf16": {}, "fp32": {}}
    for key, tiles in TILES.items():
        cls, C = key[0], key[1]
        bucket = key[2] if len(key) > 2 else "*"
        tiles = dict(tiles); modes_ok = tiles.pop("modes", ("exact", "fast"))
        if mode not in modes_ok:
            continue                      # a bucket entry measured in one mode only is never served in the other
        tab[cls].setdefault("C%d" % C, {})[bucket] = dict(V6_CFG, **tiles, **MODES[mode])
    tab["bf16"]["*"] = {"*": dict(V6_CFG, **TILES[("bf16", 256)], **MODES[mode])}
    tab["fp32"]["*"] = {"*": dict(V6_CFG, **TILES[("fp32", 384)], **MODES[mode])}
    return tab
CONFIG_TABLE = _build_config_table(_MODE)


def smem_bytes(stage, elem, dual_x):
    """Pipelined-loop shared memory estimate for A' (dual_x=False: x + 2 weight tiles) / C' (dual_x=True: 2 x + 2 weight tiles)."""
    xt = stage["BM"] * stage["BK"] * (2 if dual_x else 1); wt = 2 * stage["BK"] * stage["BN"]
    return (xt + wt) * elem * stage["num_stages"]


def _probe_smem_cap():
    """The opt-in dynamic shared memory per block of the PROBED device (H100: 232,448 B = 227 KB; A100: 166,912 B = 163 KB) -- the budget the
    tile guards below compare against.  Read from torch's device properties, else the Triton driver; the sm_90 figure only when no device is
    visible (an import on a CPU host).  A memory-dependent choice derives from the device, never from an architecture name."""
    if not torch.cuda.is_available():
        return 227 * 1024
    v = getattr(torch.cuda.get_device_properties(torch.cuda.current_device()), "shared_memory_per_block_optin", None)
    if v:
        return int(v)
    try:
        from triton.runtime import driver as _drv
        return int(_drv.active.utils.get_device_properties(torch.cuda.current_device())["max_shared_mem"])
    except (ImportError, AttributeError, KeyError, RuntimeError):
        return 227 * 1024


SMEM_CAP = _probe_smem_cap()   # 232,448 B on sm_90 (every cfg this table selects is unchanged there); 166,912 B on sm_80: the N >= 4096 tall tiles drop
                               # one pipelining stage instead of requesting 221 KB (a launch the part refuses); every selection below 4096 tokens is unchanged
GRID_SCHEDULE = ((64, {}), (128, {}), (256, {"num_warps": 8}), (512, {"num_warps": 8}))   # stage-A/C tile heights the schedule steps through as N grows, each with the
                                                                                              # launch settings a tile of that height takes (8 warps from 256 rows); the smallest
                                                                                              # height whose grid fits serves
BN_FLOOR = 32                                                                                 # a scheduled tile keeps its accumulator near the table tile's size: BN shrinks as BM
                                                                                              # grows (BN_table * BM_table / BM), never below 32 output columns (the v=14 projection's
                                                                                              # BN spans p|g pairs: its floor is 64) — H100 sweep, bf16 C=128 / C=256, N 2048..4096


def _scheduled_tile(tile, BM, settings, elem, dual_x):
    """The table `tile` at height BM: BN by the accumulator rule (BN_FLOOR), the height's launch `settings` (num_warps never below the tile's own),
    the pipelining depth reduced only if the taller tile's shared memory would exceed SMEM_CAP.  v / BK — the arithmetic — are the tile's."""
    floor = BN_FLOOR * (2 if tile.get("v", 2) == 14 else 1)
    wide = dict(tile, BM=BM, BN=min(tile["BN"], max(tile["BN"] * tile["BM"] // BM, floor)), **settings)
    wide["num_warps"] = max(tile["num_warps"], wide["num_warps"])
    while wide["num_stages"] > 1 and smem_bytes(wide, elem, dual_x) > SMEM_CAP:
        wide["num_stages"] -= 1
    return wide


def _schedule_grid(cfg, N, elem):
    """Tile-height schedule by N: a stage whose launch grid (``kernels.stage_grid_y``) would exceed CUDA's axis-1 limit at this N is served by the
    next tile height of GRID_SCHEDULE that fits (64 -> 128 -> 256 -> 512 rows, :func:`_scheduled_tile`).  Only the blocking changes (BM, BN,
    num_warps, and the depth under the shared-memory cap) — every output element's arithmetic (the BK-step K loop, fp32 accumulate, the
    epilogue) is the tile table's, so a scheduled tile computes the same bits as the table's tile.  Below a stage's first limit (N * ceil(N / BM)
    programs for stage A, ceil(N*N / BM) for the ``_k_out9`` stage C: N <= 2047 with the 64-row tiles, N <= 2849 / 2896 with the 128-row tiles)
    its tile is the table's unchanged; where no height fits, the cfg is returned unscheduled and the caller's gate refuses the geometry by name
    (``launch_grid_y>65535``)."""
    for stage, dual_x in (("A", False), ("C", True)):
        tile = cfg[stage]
        if K.stage_grid_y(stage, N, tile) <= K.CUDA_GRID_Y_MAX:
            continue
        for BM, settings in GRID_SCHEDULE:
            if BM <= tile["BM"]:
                continue
            wide = _scheduled_tile(tile, BM, settings, elem, dual_x)
            if K.stage_grid_y(stage, N, wide) <= K.CUDA_GRID_Y_MAX:
                cfg[stage] = wide
                break
    return cfg


def _select_cfg(dtype, N, C=256):
    cls = "bf16" if dtype in (torch.bfloat16, torch.float16) else "fp32"
    tab = CONFIG_TABLE[cls]
    sub = tab.get("C%d" % C, tab["*"])
    cfg = None
    for k, v in sub.items():
        if k == "*":
            cfg = cfg or v
        else:
            lo, hi = [int(x) for x in k.split("-")]
            if lo <= N <= hi: cfg = v
    cfg = dict(cfg)
    elem = 2 if cls == "bf16" else 4
    # guard (never OOM in an engine): shrink to the smallest pipelined tile if the estimate exceeds the H100 cap
    if smem_bytes(cfg["A"], elem, False) > SMEM_CAP: cfg["A"] = dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
    if smem_bytes(cfg["C"], elem, True) > SMEM_CAP: cfg["C"] = dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
    return _schedule_grid(cfg, N, elem)


# FPF_META declaration: per op, LN mode, per-stage kernel by name, byte-weighted z-passes removed.
# Stages (named S_in | A' | B | S_out | C'): S_in = our _k_ln_rows (LN_in materialised, 1 read of z + 1 write of x_in, same arithmetic as cuEq's
# layer_norm_transpose = cuEq's own Triton LN, NOT the model's fast_layernorm — cuEq TriMul never calls fast_layernorm); A' = our _k_proj6 (pure dual gated GEMM on x_in,
# channel-major padded store); B = cuBLAS bmm on the padded planes (contraction; Triton _k_contract = control, bit-exact-equal at 356/705);
# S_out = our _k_ln_cols_T (LN_out over d, transposed store = cuEq layer_norm_transpose dbij->bijd); C' = our _k_out6 (pure dual gated GEMM + residual epilogue).
# z-passes (byte-weighted, N x N x 256 bf16 = 1 unit, vs stock cuEq module forward = z.clone [2] + LN_in [2] + gated GEMM [1 read + 2 write] + einsum [2 read + 1 write]
# + LN_out transpose [2] + out GEMM [2 read + 1 write] + z + z_in [3] = 18 units): ours = S_in [2] + A' [1 + 2] + B [2 + 1] + S_out [2] + C' [2 read + 1 read z + 1 write] = 14 units
# -> z_passes_removed = 4 (clone read+write, residual read+write); launches 7 -> 7 (stock: clone, LN, GEMM, einsum, LN, GEMM, add; ours: LN, GEMM, bmm, LN, GEMM... = 5) -> launches removed 2.
def _meta(mode):
    exact = mode == "exact"
    return {
        "mode": mode,
        "ln_mode": "stock-call" if exact else "in-kernel",
        "ln_class": ("the stock cuEq layer_norm_transpose kernels are called as-is (bitwise)" if exact else
                     "own two-pass fp32 LN passes (_k_ln_rows / _k_ln_cols_T): 1 bf16 ulp vs cuEq's LN tree in ~1e-3 of elements (TIER2)") + " — not the model's fast_layernorm (the stock cuEq path never calls it)",
        "stages": ("S_in:cuEq-layer_norm_transpose|A':fpf_trimul._k_proj9|B:cuBLAS-bmm-XS=N-planes|S_out:cuEq-layer_norm_transpose|C':fpf_trimul._k_out9" if exact else
                   "S_in:fpf_trimul._k_ln_rows|A':fpf_trimul._k_proj9|B:%s|S_out:fpf_trimul._k_ln_cols_T|C':fpf_trimul._k_out9" % ("fpf_trimul._k_contract-pad64-planes" if CONTRACT == "triton" else "cuBLAS-bmm-pad8-planes")),
        "contraction_exact_vs_stock": "yes (cuBLAS on the unaligned lda=N problem = the stock einsum call; bitwise T7/T9 16/16)" if exact else
                                      "no (aligned planes change cuBLAS's accumulation order; Triton K_B is k-sequential fp32: 1/2 ulp on x; pad-INVARIANT with the Triton K_B)",
        "z_passes_removed": 4, "launches_removed": 2, "z_passes_note": "byte-weighted, N*N*C*2B = 1 unit: stock 18 units / 7 launches -> 14 / 5 (z.clone read+write and the residual read+write are fused into C's epilogue)",
        "tf32_operand_rounding": "rna (cvt.rna.tf32.f32 on both GEMM operands before every tl.dot when PREC=1, = cuBLAS/cuDNN TF32 class; program finding D78)",
        "tf32_note": "fp32 engines: A'/C' operands are the fp32 LN outputs pre-rounded RNA to TF32 in-kernel; the cuBLAS contraction rounds RNA itself; bf16 engines: not applicable",
        "label": "EXACT" if exact else "TIER2_SAME_CLASS",
    }


FPF_META = {"trimul_out": _meta(_MODE)}
FPF_META["trimul_in"] = dict(FPF_META["trimul_out"])


def config_table_sha():
    return hashlib.sha256(json.dumps(CONFIG_TABLE, sort_keys=True).encode()).hexdigest()[:16]


def _weights(module, wdt):
    cache = getattr(module, "_fpf_cache", None)
    if cache is None:
        cache = module._fpf_cache = {}
    key = ("w", wdt)
    if key not in cache:
        def c(t): return t.detach().to(wdt).contiguous()      # GEMM weights: cast exactly as cuEq does under autocast (maybe_to(autocast_dtype))
        def n(t): return t.detach().contiguous()              # LN affine params: cuEq keeps them native (loaded as fp32 inside its LN kernel)
        cache[key] = K.pack_weights(
            n(module.layer_norm_in.weight), n(module.layer_norm_in.bias),
            c(torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], 0)), c(torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], 0)),
            n(module.layer_norm_out.weight), n(module.layer_norm_out.bias), c(module.linear_z.weight), c(module.linear_g.weight))
    return cache[key]


def fn(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    """Exact stock signature.  Returns what the stock forward returns.  The stock function serves, counted BY WORD (STATS['fallback_by']): 'not-cueq' (the module does not run
    the cuEq fused TriMul with c_z == c_hidden: nothing to replace), 'rank:<d>', 'n<=100' (cuEq's own torch path), a MEASURED_OFF class ('cell:<cls>_C<C>+off(not-measured)'), a dtype
    the kernels do not compute ('dtype:<cls>'); an UNKNOWN bf16 / fp32 class is served on the default tiles and named (STATS['unverified'], ONE stderr line)."""
    import fpf
    stock = fpf.original("trimul_out" if module._outgoing else "trimul_in")
    if not (triangle_multiplicative == "cuequivariance" and module.c_z == module.c_hidden):
        _count_stock("not-cueq")
        return stock(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    if z.dim() not in (3, 4):
        _count_stock("rank:%d" % z.dim())
        return stock(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    kind, word = cell_verdict(torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else module.linear_a_p.weight.dtype, module.c_hidden)
    if kind in ("off", "unsupported"):
        _count_stock(word)
        return stock(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    if kind == "unverified":
        _note_unverified(word)
    STATS["calls"] += 1
    direction = "outgoing" if module._outgoing else "incoming"
    # dtype policy (cuEq): LN runs on z as given; the GEMM operands are cast to the autocast dtype when autocast is on, else to the weight dtype
    if torch.is_autocast_enabled():
        cdt = torch.get_autocast_dtype("cuda")
    else:
        cdt = module.linear_a_p.weight.dtype
    w = _weights(module, cdt)
    add = bool(inplace_safe is True and _add_with_inplace)
    zs = z if z.dim() == 4 else z[None]
    ms = None if mask is None else (mask if mask.dim() == 3 else mask[None])
    outs = []
    for b in range(zs.shape[0]):
        zb = zs[b].contiguous()
        N = zb.shape[0]
        if N <= 100:    # cuEq falls back to its torch path below CUEQ_TRIMUL_FALLBACK_THRESHOLD: keep the stock function there
            STATS["calls"] -= 1; _count_stock("n<=100")
            return stock(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
        mb = None if ms is None else ms[b]      # module-level mask=None -> stock passes ones (x1.0 exact) -> we skip the multiply; a real mask is applied as stock does
        cfg = _select_cfg(cdt, N, zb.shape[-1])
        outs.append(K.trimul_forward(zb, direction, mb, w, eps=1e-5, residual=add, cfg=cfg, contract=CONTRACT, cdt=cdt))
    out = torch.stack(outs, 0) if z.dim() == 4 else outs[0]
    return out
