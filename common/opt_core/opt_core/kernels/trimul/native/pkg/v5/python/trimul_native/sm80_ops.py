"""trimul_native.sm80_ops -- op assembly of the sm_80 (Ampere) member: K1 cubin | one NT batched GEMM (torch.bmm) | K3 cubin.

    from trimul_native.sm80_ops import serve_sm80
    out = serve_sm80(z, mask, direction="outgoing", weights=w10, residual=False, cache={})

z [B,N,N,c_z] or [N,N,c_z] (bf16, or fp32 = the fp32-resident pair tensor read natively), mask [B,N,N] / [N,N] 0/1 float or None, weights = the
ten canonical fp32 tensors (ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og), residual -> z + update.  Returns the update in the
compute dtype (bf16) or, with residual onto an fp32 z, the fp32 sum; z's rank.  Served widths = the (c_z, c_hidden, z dtype) keys of TILES (a tuning TABLE of kernel instantiation names; every
entry runs the same two kernel bodies); anything else steps aside BY NAME (Refusal .reason/.kind), never silently.
Planes: a | b channel-major bf16 [2*B*c_hidden][Np][Np], Np = ceil16(N), pads written as zeros by K1 (incoming: transposed planes), X = bmm(a, b^T)
bf16 [B*c_hidden][Np][Np]; numerics = the fpf_trimul_v4 rounding sequence (see csrc/sm80/math_sm80.cuh).
"""
import os
import re

import torch

from . import launch as L

__all__ = ["serve_sm80", "serve_sm80_exact", "Refusal", "TILES", "TILES_EXACT", "pack_weights", "variant_geometry"]

from .face import Refusal                                # the package's refusal-by-name type (.reason / .kind carry the word)


def _refuse(reason):
    raise Refusal(reason)


# (c_z, c_hidden, z dtype) -> (K1 instantiation, K3 instantiation).  Overridable per call through cache['_sm80_tiles'] = (k1_name, k3_name)
# or the environment (TRIMUL_SM80_K1 / TRIMUL_SM80_K3) for tuning sweeps.
def _rule_tiles(C, D, zt):
    """default table rows for a (c_z, c_hidden, z dtype) class: the K1 tile follows c_z, the K3 tile follows max(c_z, c_hidden) (register / smem
    budget of the LayerNorm-output fragments and the pre-LayerNorm tile).  Tuning data, not code paths: every name is the same two kernel bodies."""
    k1 = {64: ("bm128_bn32_wm32_s2_mb4", "bm128_bn16_wm32_s2_mb4"), 128: ("bm128_bn32_wm32_s2_mb3",) * 2,
          256: ("bm128_bn32_wm32_s2_mb2",) * 2, 384: ("bm64_bn16_wm16_s2_mb2",) * 2}[C][0 if zt == "bf16" else 1]
    if D == 384 or C == 384: k3 = "bm64_bn16_wm16_s1_mb2"
    elif D == 256 or C == 256: k3 = "bm64_bn16_wm32_s1_mb4"
    elif C == 128: k3 = "bm128_bn32_wm32_s2_mb2" if zt == "bf16" else "bm64_bn32_wm32_s1_mb4"
    else: k3 = "bm128_bn32_wm32_s2_mb4"
    p1, p3 = ("k1", "k3") if zt == "bf16" else ("k1z", "k3p")                     # fp32 z: K1 also writes its LayerNorm rows, K3 reads them (not z)
    return ("%s_%s_c%d_h%d_%s" % (p1, zt, C, D, k1), "%s_%s_c%d_h%d_%s" % (p3, zt, C, D, k3))


TILES = {
    (128, 128, "bf16"): ("k1_bf16_c128_h128_bm128_bn32_wm32_s2_mb3", "k3_bf16_c128_h128_bm128_bn32_wm32_s2_mb2"),
    (128, 128, "f32"):  ("k1z_f32_c128_h128_bm128_bn32_wm32_s2_mb3",  "k3p_f32_c128_h128_bm128_bn32_wm32_s1_mb2"),
    (256, 256, "bf16"): ("k1_bf16_c256_h256_bm128_bn32_wm32_s2_mb2", "k3_bf16_c256_h256_bm64_bn32_wm16_s1_mb2"),
    (256, 256, "f32"):  ("k1z_f32_c256_h256_bm128_bn32_wm32_s2_mb2",  "k3p_f32_c256_h256_bm64_bn32_wm16_s1_mb2"),
    (64, 64, "bf16"):   ("k1_bf16_c64_h64_bm128_bn32_wm32_s2_mb4",   "k3_bf16_c64_h64_bm64_bn64_wm16_s1_mb4"),
    (64, 64, "f32"):    ("k1z_f32_c64_h64_bm128_bn16_wm32_s2_mb4",    "k3p_f32_c64_h64_bm128_bn16_wm32_s2_mb4"),
    (64, 128, "bf16"):  ("k1_bf16_c64_h128_bm128_bn32_wm32_s2_mb4",  "k3_bf16_c64_h128_bm64_bn64_wm16_s1_mb3"),
    (64, 128, "f32"):   ("k1z_f32_c64_h128_bm128_bn16_wm32_s2_mb4",   "k3p_f32_c64_h128_bm128_bn16_wm32_s2_mb4"),
    (384, 384, "bf16"): ("k1_bf16_c384_h384_bm64_bn16_wm16_s2_mb3",  "k3_bf16_c384_h384_bm64_bn32_wm16_s1_mb1"),
    (384, 384, "f32"):  ("k1z_f32_c384_h384_bm64_bn16_wm16_s2_mb2",   "k3p_f32_c384_h384_bm64_bn32_wm16_s1_mb1"),
}
for _C in (64, 128, 256, 384):
    for _D in (64, 128, 256, 384):
        for _zt in ("bf16", "f32"):
            TILES.setdefault((_C, _D, _zt), _rule_tiles(_C, _D, _zt))
# bit-exact rows (== the stock library bit for bit; c_z = c_hidden = 256, bf16 pair tensor): K1 with the stock-order LayerNorm, K3 keyed by N % 4
TILES_EXACT = {(256, 256, "bf16"): ("k1x_bf16_c256_h256_bm128_bn32_wm32_s2_mb2", "k3x1_bf16_c256_h256_bm64_bn32_wm16_s1_mb2", "k3x2_bf16_c256_h256_bm64_bn32_wm16_s1_mb2"),
               (64, 64, "bf16"):   ("k1x_bf16_c64_h64_bm128_bn32_wm32_s2_mb4",   "k3x1_bf16_c64_h64_bm64_bn64_wm16_s1_mb4",    "k3x2_bf16_c64_h64_bm64_bn64_wm16_s1_mb4"),
               (128, 128, "bf16"): ("k1x_bf16_c128_h128_bm128_bn32_wm32_s2_mb3", "k3x1_bf16_c128_h128_bm128_bn32_wm32_s2_mb2", "k3x2_bf16_c128_h128_bm128_bn32_wm32_s2_mb2"),
               (384, 384, "bf16"): ("k1x_bf16_c384_h384_bm64_bn16_wm16_s2_mb3",  "k3x1_bf16_c384_h384_bm64_bn32_wm16_s1_mb1",  "k3x2_bf16_c384_h384_bm64_bn32_wm16_s1_mb1")}
_NAME_RE = re.compile(r"^k([13])(x[12]?|z|p|t)?_(bf16|f32)_c(\d+)_h(\d+)_bm(\d+)_bn(\d+)_wm(\d+)_s(\d+)_mb(\d+)$")
PAD = 16
EPS = 1e-5


def variant_geometry(name):
    """-> dict(kind, zt, C, D, BM, BN, WM, STAGES, MB, threads, smem) from an instantiation name (the smem formulas of the .cu files)."""
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError("bad sm80 kernel name %r" % name)
    kind, xtag, zt = int(m.group(1)), (m.group(2) or ""), m.group(3)
    C, D, BM, BN, WM, ST, MB = (int(v) for v in m.groups()[3:])
    NW = BM // WM
    esz = 2 if zt == "bf16" else 4
    if kind == 1:
        smem = ST * 2 * BN * C * 2 + 2 * BN * (BM + 8) * 2
    else:
        smem = max(D * BM, BM * C) * 2 + ST * BN * (D + C) * 2 + NW * WM * (BN + 8) * 2
    stash = BM * C * 2 if (kind == 3 and zt == "bf16") else 0                     # optional raw-z tile behind the staging area for the residual add
    # residual onto an fp32 z: K3 stages [BM][BN] fp32 sums (z chunk cp.async-ed in, o added in place) instead of [BM][BN+8] bf16 o -> larger staging
    res32 = BM * ((BN + (8 if BN % 32 == 0 else 4)) * 4 - (BN + 8) * 2) if (kind == 3 and zt == "f32") else 0
    return dict(kind=kind, zt=zt, C=C, D=D, BM=BM, BN=BN, WM=WM, STAGES=ST, MB=MB, threads=NW * 32, smem=smem, name=name, exact=xtag, stash_bytes=stash, res32_extra=res32)


WORKSPACE_PAIR_KEEP_BYTES = 256 << 20                # the (unit, Np) planes + contraction-output pair stays cached only up to this total
_KEEP_FRACTION = {"ab": 4.0 / 6.0, "x": 2.0 / 6.0, "zln": 2.0 / 6.0}


def workspace_kept(kind, nbytes):
    """Same workspace policy as the sm_90a op assembly: a buffer of role ``kind`` is kept in the caller's cache iff it is within its share of
    WORKSPACE_PAIR_KEEP_BYTES (planes 4/6, contraction output and LayerNorm rows 2/6); above that it is a transient allocation from the
    framework's caching allocator and nothing but weight-derived state stays resident after a call."""
    return nbytes <= _KEEP_FRACTION.get(kind, 2.0 / 6.0) * WORKSPACE_PAIR_KEEP_BYTES


def pack_weights(w, device):
    """The ten canonical fp32 tensors -> the kernels' operands (bf16 round-to-nearest-even casts = the autocast rounding of the stock op)."""
    C = int(w["w_ag"].shape[1]); D = int(w["w_ag"].shape[0])
    f = lambda t: t.detach().to(device=device, dtype=torch.float32).contiguous()  # noqa: E731
    wg = torch.cat([f(w["w_ag"]), f(w["w_bg"])], 0).to(torch.bfloat16).contiguous()
    wp = torch.cat([f(w["w_ap"]), f(w["w_bp"])], 0).to(torch.bfloat16).contiguous()
    return dict(C=C, D=D, wg=wg, wp=wp, wo=f(w["w_o"]).to(torch.bfloat16).contiguous(), wog=f(w["w_og"]).to(torch.bfloat16).contiguous(),
                g_in=f(w["ln_in_w"]), b_in=f(w["ln_in_b"]), g_out=f(w["ln_out_w"]), b_out=f(w["ln_out_b"]))


_KERNELS = {}


SMEM_SM = 166912                                                                 # opt-in dynamic shared memory per block on cc 8.0 (bytes)


def _ctas_per_sm(k, dyn_smem, smem_sm=166912, regs_sm=65536, threads_sm=2048, blocks_sm=32):
    """resident CTAs per SM of kernel k launched with dyn_smem bytes (cc 8.0 limits: 164 KB shared memory with 1 KB per-CTA reserve, 64K
    registers allocated per warp in units of 256, 2048 threads, 32 CTAs)."""
    thr = k.geo["threads"]; warps = thr // 32
    regs = max(int(getattr(k, "regs", 0) or 0), 1)
    warp_regs = -(-regs * 32 // 256) * 256
    by_regs = regs_sm // (warp_regs * warps) if warp_regs * warps else blocks_sm
    by_smem = smem_sm // (int(dyn_smem) + int(getattr(k, "static_smem", 0) or 0) + 1024)
    return max(0, min(by_smem, by_regs, threads_sm // thr, blocks_sm))


def _kernel(name, device_index):
    key = (name, device_index)
    k = _KERNELS.get(key)
    if k is None:
        geo = variant_geometry(name)
        unit = L.load_unit("trimul_k1_sm80" if geo["kind"] == 1 else "trimul_k3_sm80", device=device_index)
        k = unit.kernel(name)
        extra = max(geo.get("stash_bytes", 0), geo.get("res32_extra", 0))           # optional regions: raw-z tile (bf16 residual) | fp32 sum staging (fp32 residual)
        want = geo["smem"] + extra if geo["smem"] + extra + 1024 <= SMEM_SM else geo["smem"]
        k.set_max_dynamic_smem(want); k._dyn_smem_max = want
        k.geo = geo
        _KERNELS[key] = k
    return k


def _launch(k, grid, block, args, smem):
    """launch with the dynamic shared memory this call needs; the opt-in attribute is raised first if a launch asks for more than was set at
    load (residual staging regions, rows pinned through overrides), so no launch can fail on the attribute."""
    if smem > getattr(k, "_dyn_smem_max", 48 * 1024):
        if smem + 1024 > SMEM_SM:
            _refuse("tile:%s needs %d B of shared memory" % (k.geo["name"], smem))
        k.set_max_dynamic_smem(smem); k._dyn_smem_max = smem
    k.launch_packed(grid, block, args, smem=smem)


# Contraction FORM.  The planes can be written in either token order by the same K1 body (its `outgoing` flag says whether plane (row r,
# col c) holds token (r, c) or (c, r)), so X = sum_k a[i,k] b[j,k] is served either as  NT: X = A . B^T on planes [i][k]  or as
# TN: X = P^T . Q on planes [k][i].  cuBLAS (torch.bmm) runs visibly different kernels for the two; on cc 8.0 (A100-80GB, torch 2.13.0+cu130)
# the TN form measured 1.5-4 % faster at c_hidden 64 | 128 for every Np in {400, 800, 1200, 1536, 2048} (0.962-0.985 of NT) and produced
# byte-identical op outputs, so TN is the sm_80 default; TRIMUL_SM80_FORM=NT|TN or cache['_sm80_form'] overrides (tuning data, not a code
# path: both directions x both forms run the same two kernels).
FORM_DEFAULT = "TN"


def _form(cache=None):
    f = os.environ.get("TRIMUL_SM80_FORM") or (cache or {}).get("_sm80_form") or FORM_DEFAULT
    return "NT" if f.upper() == "NT" else "TN"


def _tiles_for(C, D, zt, cache):
    over = (cache or {}).get("_sm80_tiles") if cache is not None else None
    k1n = os.environ.get("TRIMUL_SM80_K1") or (over[0] if over else None)
    k3n = os.environ.get("TRIMUL_SM80_K3") or (over[1] if over else None)
    base = TILES.get((C, D, zt))
    if base is None and not (k1n and k3n):
        _refuse("shape:c_z=%d,c_hidden=%d,%s(no sm_80 tile row)" % (C, D, zt))
    k1n = k1n or base[0]; k3n = k3n or base[1]
    for n, kind in ((k1n, 1), (k3n, 3)):
        g = variant_geometry(n)
        if g["kind"] != kind or g["C"] != C or g["D"] != D or g["zt"] != zt:
            _refuse("tile:%s does not serve (c_z=%d, c_hidden=%d, %s)" % (n, C, D, zt))
    return k1n, k3n


def _prep_mask(mask, B, N):
    if mask is None:
        return None, 0
    m = mask
    if m.dtype != torch.float32:
        m = m.to(torch.float32)
    if m.dim() == 2:
        mbs = 0
    elif m.dim() == 3:
        mbs = 0 if m.shape[0] == 1 else N * N
        if m.shape[0] not in (1, B):
            _refuse("shape:mask%s" % (tuple(mask.shape),))
    else:
        _refuse("shape:mask%s" % (tuple(mask.shape),))
    return (m if m.is_contiguous() else m.contiguous()), mbs


def serve_sm80(z, mask=None, *, direction, weights, residual=False, cache=None, eps=EPS, exact=False):
    """exact=True: the bit-exact variant (torch.equal to cuequivariance_torch 0.11 triangle_multiplicative_update under bf16 autocast): c_z = c_hidden
    in {64, 128, 256, 384}, bf16 pair tensor, batch 1; other classes step aside by name ("exact:...")."""
    if cache is None:
        cache = {}
    if not z.is_cuda:
        _refuse("device:cpu")
    cc = torch.cuda.get_device_capability(z.device)
    if cc[0] != 8:
        _refuse("cc:%d.%d!=8.x(sm_80 binary)" % cc)
    if z.dtype not in (torch.bfloat16, torch.float32):
        _refuse("dtype:%s" % str(z.dtype).replace("torch.", ""))
    zt = "bf16" if z.dtype == torch.bfloat16 else "f32"
    squeeze = z.dim() == 3
    z4 = z.unsqueeze(0) if squeeze else z
    if z4.dim() != 4 or z4.shape[1] != z4.shape[2]:
        _refuse("shape:z%s" % (tuple(z.shape),))
    B, N, _, C = (int(v) for v in z4.shape)
    dname = direction if isinstance(direction, str) else ("outgoing" if direction else "incoming")
    if dname not in ("outgoing", "incoming"):
        _refuse("direction:%s" % dname)
    # weights: packed once per weight SET, keyed by the source tensors' storage (a cache shared by several layers holds one pack per layer; the
    # by-value launch structs below carry the pack's pointers, so they are keyed by the same signature and the pack is kept alive with them)
    wsrc = tuple(int(weights[k].data_ptr()) for k in ("w_ag", "w_ap", "w_bg", "w_bp", "w_o", "w_og", "ln_in_w", "ln_in_b", "ln_out_w", "ln_out_b"))
    wsrc = wsrc + tuple((tuple(weights[k].shape), str(weights[k].dtype)) for k in ("w_ag", "w_o", "ln_in_w", "ln_out_w"))   # address / shape / dtype (inference tensors carry no version counter)
    pk = cache.get(("_sm80_w", wsrc))
    if pk is None:
        pk = pack_weights(weights, z.device); pk["_src"] = wsrc; cache[("_sm80_w", wsrc)] = pk
    if pk["C"] != C:
        _refuse("shape:c_z=%d(weights c_z=%d)" % (C, pk["C"]))
    D = pk["D"]
    dev = z.device.index if z.device.index is not None else torch.cuda.current_device()
    if not z4.is_contiguous():
        z4 = z4.contiguous()
    if exact:
        return _serve_exact(z4, mask, squeeze, B, N, C, D, zt, dname, pk, residual, cache, eps, dev)
    k1n, k3n = _tiles_for(C, D, zt, cache)
    use_stash = os.environ.get("TRIMUL_SM80_STASH", "1") != "0"
    if os.environ.get("TRIMUL_SM80_PRELN", "1") == "0":                           # dev A/B switch: fall back to the rows that re-normalise z in K3
        k1n, k3n = k1n.replace("k1z_", "k1_", 1), k3n.replace("k3p_", "k3_", 1)
    if os.environ.get("TRIMUL_SM80_SIGMOID", "") == "tanh" and k1n.startswith("k1_") and k3n.startswith("k3_"):   # measurement rows outside the acceptance class
        k1n, k3n = "k1t_" + k1n[3:], "k3t_" + k3n[3:]
    preln = k1n.startswith("k1z_")                                                 # K1 also writes its LayerNorm rows (bf16); K3 reads those instead of z
    if preln != k3n.startswith("k3p_"):
        _refuse("tile:%s + %s (a LayerNorm-rows-out K1 row pairs with a rows-in K3 row)" % (k1n, k3n))
    k1 = _kernel(k1n, dev); k3 = _kernel(k3n, dev)
    m, mbs = _prep_mask(mask, B, N)
    g1, g3 = k1.geo, k3.geo
    ev = None
    if os.environ.get("TRIMUL_SM80_STAGE_TIMES"):                                 # dev instrument: per-stage device times -> cache['_stages_ms']
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]; ev[0].record()
    Np = (N + PAD - 1) // PAD * PAD
    bkey = ("_sm80_buf", B, D, Np, N, C, preln, dev)
    keep = workspace_kept("ab", 2 * B * D * Np * Np * 2) and workspace_kept("x", B * D * Np * Np * 2) and (not preln or workspace_kept("zln", B * N * N * C * 2))
    bufs = cache.get(bkey) if keep else None
    if bufs is None:                                                               # plane / GEMM-output scratch: cached (one shape per cache) while the pair is small,
        for k_ in [k_ for k_ in cache if isinstance(k_, tuple) and k_ and k_[0] == "_sm80_buf"]:   # a transient allocation from the caching allocator above the threshold
            del cache[k_]
        bufs = (torch.empty((2 * B * D, Np, Np), dtype=torch.bfloat16, device=z.device), torch.empty((B * D, Np, Np), dtype=torch.bfloat16, device=z.device),
                torch.empty((B, N, N, C), dtype=torch.bfloat16, device=z.device) if preln else None)
        if keep:
            cache[bkey] = bufs
    ab, x, zln = bufs
    # output dtype follows the stock op: the update in the compute dtype (bf16); the residual sum onto an fp32 pair tensor in fp32
    out = torch.empty(z4.shape, dtype=(torch.float32 if (residual and zt == "f32") else torch.bfloat16), device=z.device)
    # launch records: the parameter structs are packed ONCE per (kernel pair, shape, form) and only the tensor addresses are patched per call
    # residual onto a bf16 z: keep the raw z tile in shared memory for the add when the row's residency allows it (else z is re-read there)
    ctas0 = _ctas_per_sm(k3, g3["smem"])                                           # CTAs per SM without / with the raw-z tile (shared memory, registers, threads)
    ctas1 = _ctas_per_sm(k3, g3["smem"] + g3.get("stash_bytes", 0))
    stash = 1 if (residual and zt == "bf16" and g3.get("stash_bytes") and use_stash and ctas1 >= ctas0) else 0   # only when residency is unchanged (else the z chunk is staged per weight chunk)
    if os.environ.get("TRIMUL_SM80_STASH") == "2" and residual and zt == "bf16" and g3.get("stash_bytes") and g3["smem"] + g3["stash_bytes"] + 1024 <= SMEM_SM:
        stash = 1                                                                  # dev switch: the raw-z tile even where residency drops
    smem3 = g3["smem"] + (g3["stash_bytes"] if stash else 0) + (g3.get("res32_extra", 0) if (residual and zt == "f32") else 0)
    if smem3 + 1024 > SMEM_SM:
        _refuse("tile:%s residual staging %d B exceeds shared memory" % (k3n, smem3))
    form = _form(cache)
    rowtok = (dname == "outgoing") != (form == "TN")                              # plane (r, c) <- token (r, c) [True] or token (c, r) [False]
    pkey = ("_sm80_packs", wsrc, k1n, k3n, B, N, C, D, zt, dname, form, mbs, m is not None, bool(residual), stash, float(eps), dev)   # every by-value scalar of the structs is in the key
    packs = cache.get(pkey)
    if packs is None:
        zl = zln if zln is not None else L.u64(0)
        p1 = L.Struct([z4, (m if m is not None else L.u64(0)), ab, pk["wg"], pk["wp"], pk["g_in"], pk["b_in"], zl,
                       L.i64(N * N * C), L.i64(mbs), L.i64(Np * Np), L.i32(N), L.i32(Np), L.i32(B), L.i32(D), L.i32(1 if rowtok else 0), L.f32(eps)])
        p3 = L.Struct([x, z4, out, pk["wo"], pk["wog"], pk["g_out"], pk["b_out"], pk["g_in"], pk["b_in"], zl,
                       L.i64(N * N * C), L.i64(Np * Np), L.i32(N), L.i32(Np), L.i32(B), L.i32(D), L.i32(1 if residual else 0), L.i32(stash), L.f32(eps)])
        packs = (k1.argpack([p1]), k3.argpack([p3]),
                 ((Np + g1["BM"] - 1) // g1["BM"], Np, B), (g1["threads"], 1, 1), ((N + g3["BM"] - 1) // g3["BM"], N, B), (g3["threads"], 1, 1))
        cache[pkey] = packs
    a1, a3, grid1, blk1, grid3, blk3 = packs
    a1.set_ptr(0, z4, field=0); a1.set_ptr(0, (m if m is not None else 0), field=1); a1.set_ptr(0, ab, field=2)
    a3.set_ptr(0, x, field=0); a3.set_ptr(0, z4, field=1); a3.set_ptr(0, out, field=2)
    if zln is not None:                                                            # the LayerNorm-rows buffer is a workspace too: patch its address per call
        a1.set_ptr(0, zln, field=7); a3.set_ptr(0, zln, field=9)
    _launch(k1, grid1, blk1, a1, g1["smem"])
    if ev: ev[1].record()
    if form == "TN":
        torch.bmm(ab[:B * D].transpose(1, 2), ab[B * D:], out=x)                 # X[p, i, j] = sum_k P[p, k, i] Q[p, k, j]  (planes [k][i]; both directions)
    else:
        torch.bmm(ab[:B * D], ab[B * D:].transpose(1, 2), out=x)                 # X[p, i, j] = sum_k a[p, i, k] b[p, j, k]  (planes [i][k]; both directions)
    if ev: ev[2].record()
    _launch(k3, grid3, blk3, a3, smem3)
    if ev:
        ev[3].record(); ev[3].synchronize()
        cache["_stages_ms"] = {"k1": ev[0].elapsed_time(ev[1]), "bmm": ev[1].elapsed_time(ev[2]), "k3": ev[2].elapsed_time(ev[3])}
    cache["_describe"] = "trimul_native sm_80 member: %s | torch.bmm %s | %s" % (k1n, form, k3n)
    cache["_cfg"] = {"k1": k1n, "k3": k3n, "Np": Np, "preln": preln, "stash": stash, "form": form}
    return out.squeeze(0) if squeeze else out


def _serve_exact(z4, mask, squeeze, B, N, C, D, zt, dname, pk, residual, cache, eps, dev):
    """Bit-exact assembly: K1 (stock-order LN_in, stock gating arithmetic) -> UNPADDED planes [2*c_h, N, N] in the (i, k) orientation for both
    directions -> the stock contraction expression on [c_h, 1, N, N] chunk views (hence the stock cuBLAS problem, kernel and bits) -> K3 with the
    stock transposing-LayerNorm order (two summation trees, selected by N % 4) and the module's residual add."""
    key = (C, D, zt)
    if zt != "bf16":
        _refuse("exact:dtype=f32(the bit-exact rows read a bf16 pair tensor; cast first = the stock op's own first step)")
    if key not in TILES_EXACT:
        _refuse("exact:c_z=%d,c_hidden=%d,%s(bit-exact rows exist for c_z=c_hidden in 64|128|256|384, bf16)" % key)
    if B != 1:
        _refuse("exact:batch=%d(the stock plane layout is replicated for batch 1)" % B)
    k1n, k3a, k3b = TILES_EXACT[key]
    xo = os.environ.get("TRIMUL_SM80_K3X")                                          # tuning sweeps: "k3x1_name,k3x2_name" (or cache['_sm80_tiles_exact'])
    xo = tuple(xo.split(",")) if xo else cache.get("_sm80_tiles_exact")
    if xo:
        k3a, k3b = xo[0], xo[1]
    k3n = k3a if N % 4 == 0 else k3b
    k1 = _kernel(k1n, dev); k3 = _kernel(k3n, dev)
    g1, g3 = k1.geo, k3.geo
    m, mbs = _prep_mask(mask, B, N)
    Np = N if N % 8 == 0 else (N + PAD - 1) // PAD * PAD                            # 16-byte plane rows: direct when N % 8 == 0, else padded + one pack copy
    ab = torch.empty((2 * D, Np, Np), dtype=torch.bfloat16, device=z4.device)
    p1 = L.Struct([z4, (m if m is not None else L.u64(0)), ab, pk["wg"], pk["wp"], pk["g_in"], pk["b_in"], L.u64(0),
                   L.i64(N * N * C), L.i64(mbs), L.i64(Np * Np), L.i32(N), L.i32(Np), L.i32(B), L.i32(D), L.i32(1), L.f32(eps)])
    _launch(k1, ((Np + g1["BM"] - 1) // g1["BM"], Np, B), (g1["threads"], 1, 1), k1.argpack([p1]), g1["smem"])
    if Np != N:
        ab = ab[:, :N, :N].contiguous()
    a, b_ = torch.chunk(ab.reshape(2 * D, 1, N, N), 2, dim=0)
    x = torch.einsum("dbik,dbjk->dbij", a, b_) if dname == "outgoing" else torch.einsum("dbki,dbkj->dbij", a, b_)
    x = x.reshape(D, N, N)
    NpX = N
    if N % 8:                                                                       # K3 reads plane rows in 16-byte pieces: one pad copy of the result
        NpX = (N + 7) // 8 * 8
        xp = torch.zeros((D, NpX, NpX), dtype=torch.bfloat16, device=z4.device)
        xp[:, :N, :N].copy_(x)
        x = xp
    out = torch.empty(z4.shape, dtype=torch.bfloat16, device=z4.device)
    p3 = L.Struct([x, z4, out, pk["wo"], pk["wog"], pk["g_out"], pk["b_out"], pk["g_in"], pk["b_in"], L.u64(0),
                   L.i64(N * N * C), L.i64(NpX * NpX), L.i32(N), L.i32(NpX), L.i32(B), L.i32(D), L.i32(1 if residual else 0), L.i32(0), L.f32(eps)])
    _launch(k3, ((N + g3["BM"] - 1) // g3["BM"], N, B), (g3["threads"], 1, 1), k3.argpack([p3]), g3["smem"])
    cache["_describe"] = "trimul_native sm_80 member, bit-exact variant: %s | stock contraction (cuBLAS) | %s" % (k1n, k3n)
    cache["_cfg"] = {"k1": k1n, "k3": k3n, "Np": Np, "exact": True}
    return out.squeeze(0) if squeeze else out


def serve_sm80_exact(z, mask=None, *, direction, weights, residual=False, cache=None, eps=EPS):
    """the bit-exact variant as a plug-in callable (bf16 pair tensor; c_z = c_hidden in {64, 128, 256, 384})"""
    return serve_sm80(z, mask, direction=direction, weights=weights, residual=residual, cache=cache, eps=eps, exact=True)


triangle_multiplication = serve_sm80
forward = serve_sm80
