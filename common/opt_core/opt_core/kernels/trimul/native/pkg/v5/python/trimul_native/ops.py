# SPDX-License-Identifier: Apache-2.0
"""Op assembly of the triangle multiplication on the trimul_native kernels:  K1 (LN_in + gated dual projection + mask -> channel-major bf16 planes
[2 c_hidden, Np, Np], Np = ceil16(N), zero pads written by K1) | one strided-batched cuBLAS GEMM on the planes (bf16 operands, fp32 accumulate, bf16
out) | K3 (LN_in recomputed for the output gate + LN_out + output projection x sigmoid(gate) [+ residual] -> [N, N, c_z] in z's dtype).

Op boundary (== the provider's torch_math):
    z [N,N,c_z] | [B,N,N,c_z] = PRE-LayerNorm pair tensor (bf16, or fp32 under bf16 compute), mask [N,N] | [B,N,N] 0/1 or None
    x = LN_in(z);  a = mask*sigmoid(x W_ag^T)*(x W_ap^T);  b = mask*sigmoid(x W_bg^T)*(x W_bp^T)
    X[i,j,:] = sum_k a[i,k,:] b[j,k,:] (outgoing) | sum_k a[k,i,:] b[k,j,:] (incoming)
    update = sigmoid(x W_og^T) * (LN_out(X) W_o^T);  returns update, or z + update when residual=True (dtype of z)
Weights: the ten canonical tensors  ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og  (dict or attribute access).

Variants:  fast (default): own-order LayerNorm, padded planes, torch.bmm;  exact=True (bf16 z): LayerNorm in the reference library's summation
order inside K1/K3, unpadded planes, the stock einsum contraction -> bit-identical to cuequivariance_torch 0.11.1 triangle_multiplicative_update at
c_z = c_hidden = 256 (trimul_tx 1.2's construction); at other widths the identity is a measured property (see CELLS).
Incoming direction: 'tn' = planes in natural layout + TN GEMM (bmm(a^T, b)); 'kt' = K1 reads z with the token axes swapped so the planes come out
transposed and the contraction is the same NT GEMM as outgoing.
"""
import os

import torch

from . import kernel as K

WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
_KERNELS = {"obj": None}
# numerics class -> (K1 LNM, K3 LNM) name fields.  "ref": K1 normalises with the reference library's summation tree (l2: the a/b planes are then
# bitwise the reference's for bf16 z -- the contraction amplifies any plane difference N-fold, so this is where exact order pays), K3 with the
# shared statement in its own balanced order (l1, free); "tx": trimul_tx 1.2's arithmetic in both (comparison only).
_LNM = {"ref": (2, 1), "tx": (4, 4)}
_CACHE = {}          # process-wide serve cache when the caller passes none (packed weights by id of the mapping, workspaces, descriptors, arg packs)


class Unsupported(RuntimeError):
    """The call is outside what the loaded kernels serve (width pair without a cubin, dtype, device capability); raised by name.
    .kind = the first token of the message (shape | dtype | no_cubin | no_table | exact_dtype | k3 ...)."""

    def __init__(self, msg, kind=None):
        super().__init__(msg)
        self.kind = kind or str(msg).split(" ")[0].split(":")[0]


def ceil16(n):
    return (n + 15) // 16 * 16


def kernels(build_dir=None):
    """The process-wide Kernels object (units are loaded lazily per (c_z, c_hidden); build_dir defaults to the package's build/ or
    $TRIMUL_NATIVE_BUILD_DIR)."""
    if _KERNELS["obj"] is None:
        _KERNELS["obj"] = K.Kernels(build_dir=build_dir)
    return _KERNELS["obj"]


def _get(w, k):
    return w[k] if isinstance(w, dict) else getattr(w, k)


def pack_weights(weights, device=None):
    """Kernel-side weight pack from the ten canonical tensors:
        w1  [4 c_h, c_z] bf16: c_h/16 blocks x [32 gate rows ; 32 proj rows] over the 2 c_h plane channels (a = w_a*, then b = w_b*), k contiguous
        wg  [c_z, c_z] bf16 = W_og (out x in);  wo [c_z, c_h] bf16 = W_o;  ln_* fp32."""
    with torch.no_grad():
        w_ag, w_ap, w_bg, w_bp = (_get(weights, k) for k in ("w_ag", "w_ap", "w_bg", "w_bp"))
        ch, cz = w_ag.shape
        dev = device or w_ag.device
        gate = torch.cat([w_ag, w_bg], 0).float()
        proj = torch.cat([w_ap, w_bp], 0).float()
        blocks = []
        for b in range(2 * ch // 32):
            blocks.append(gate[32 * b:32 * b + 32]); blocks.append(proj[32 * b:32 * b + 32])
        f32 = lambda k: _get(weights, k).detach().to(dev, torch.float32).contiguous()
        return dict(cz=int(cz), ch=int(ch),
                    w1=torch.cat(blocks, 0).to(dev, torch.bfloat16).contiguous(),
                    wg=_get(weights, "w_og").detach().to(dev, torch.bfloat16).contiguous(),
                    wo=_get(weights, "w_o").detach().to(dev, torch.bfloat16).contiguous(),
                    ln_in_w=f32("ln_in_w"), ln_in_b=f32("ln_in_b"), ln_out_w=f32("ln_out_w"), ln_out_b=f32("ln_out_b"))


def _packed(weights, cache, device):
    """The kernel-side pack of a caller's ten-tensor mapping, memoised in `cache`.  The entry PINS the source tensors: a key made of addresses is
    only sound while those addresses cannot be recycled, so the entry holds the tensors it was built from (a hit then means the same live tensors;
    a caller that rewrites weights in place at the same address must drop the entry).  Bounded: at most 64 packs per cache dict."""
    if isinstance(weights, dict) and "w1" in weights:
        return weights
    src = tuple(weights[k] for k in WEIGHT_KEYS)
    key = ("tmn_pack",) + tuple((t.data_ptr(), tuple(t.shape), t.dtype, str(t.device)) for t in src)
    if cache is not None:
        ent = cache.get(key)
        if ent is not None and all(a is b for a, b in zip(ent[1], src)):
            return ent[0]
    p = pack_weights(weights, device)
    if cache is not None:
        packs = cache.setdefault(("tmn_pack", "keys"), [])
        if len(packs) >= 64:
            for k in packs:
                cache.pop(k, None)
            del packs[:]
        packs.append(key)
        cache[key] = (p, src)
    return p


WORKSPACE_PAIR_KEEP_BYTES = 256 << 20                # the (unit, Np) planes + contraction-output pair stays cached only up to this total
_KEEP_FRACTION = {"ab": 4.0 / 6.0, "x": 2.0 / 6.0, "zcast": 2.0 / 6.0, "xz": 2.0 / 6.0}   # planes [2 ch, Np, Np] : output [ch, Np, Np] = 4 : 2 (bf16)


def workspace_kept(kind, nbytes):
    """The workspace policy: a buffer of role ``kind`` and size ``nbytes`` is kept in the caller's cache iff it is within its share of
    WORKSPACE_PAIR_KEEP_BYTES (planes 4/6, contraction output and cast rows 2/6) -- equivalently the planes + output pair of one (unit, Np) is
    persistent iff it totals <= 256 MiB; above that every workspace is a transient allocation from the framework's caching allocator, so
    the resident footprint after a call is the output alone.  Descriptors are keyed by address and re-encode when a transient block moves."""
    return nbytes <= _KEEP_FRACTION.get(kind, 2.0 / 6.0) * WORKSPACE_PAIR_KEEP_BYTES


def _buf(cache, key, shape, dtype, device):
    """Workspace from the per-call-site cache (same N -> same buffers -> same tensor maps) when the workspace policy keeps it, else a fresh
    (transient) allocation.  Buffers allocated under torch.inference_mode are inference tensors (not writable by bmm(out=) outside it), so the
    inference-mode state is part of the key."""
    nbytes = int(torch.Size(shape).numel()) * torch.empty((), dtype=dtype).element_size()
    if cache is None or not workspace_kept(str(key[0]), nbytes):
        return torch.empty(shape, dtype=dtype, device=device)
    key = key + (torch.is_inference_mode_enabled(),)
    t = cache.get(key)
    if t is None or t.shape != torch.Size(shape) or t.dtype != dtype or t.device != device:
        t = torch.empty(shape, dtype=dtype, device=device)
        cache[key] = t
    return t


# ------------------------------------------------------------------------------------------------------------------------------------------------
def planes(z3, mask2, w, *, transpose=False, lnm=1, Np=None, cfg=None, cache=None, stats=None, eps=1e-5):
    """K1: z3 [N, N, cz] -> planes [2 ch, Np, Np] bf16 (Np = ceil16(N) unless given)."""
    if cache is None:
        cache = _CACHE
    N = z3.shape[0]
    Np = ceil16(N) if Np is None else Np
    kk = kernels()
    ab = _buf(cache, ("ab", Np, w["ch"]), (2 * w["ch"], Np, Np), torch.bfloat16, z3.device)
    kk.k1(z3, mask2, w, ab, N=N, Np=Np, cz=w["cz"], ch=w["ch"], cfg=cfg, lnm=lnm, transpose=transpose, stats=stats, eps=eps, cache=cache)
    return ab


def contract(ab, ch, outgoing, planes_transposed=False, out=None):
    """X[c,i,j] = sum_k a[c,i,k] b[c,j,k] (outgoing) | sum_k a[c,k,i] b[c,k,j] (incoming): one strided-batched GEMM.  With transposed planes
    (K1 transpose=True) the incoming contraction is the NT form as well."""
    a, b = ab[:ch], ab[ch:]
    if outgoing or planes_transposed:
        return torch.bmm(a, b.transpose(1, 2), out=out) if out is not None else torch.bmm(a, b.transpose(1, 2))
    return torch.bmm(a.transpose(1, 2), b, out=out) if out is not None else torch.bmm(a.transpose(1, 2), b)


def epilogue(x, z3, w, *, residual, lnm=1, cfg=None, cache=None, out=None, eps=1e-5):
    """K3: x planes [ch, Np, Np] bf16 + z3 -> out [N, N, cz] (z's dtype)."""
    if cache is None:
        cache = _CACHE
    N = z3.shape[0]
    kk = kernels()
    out = torch.empty((N, N, w["cz"]), dtype=z3.dtype, device=z3.device) if out is None else out
    kk.k3(x, z3, w, out, N=N, Np=x.shape[-1], cz=w["cz"], ch=w["ch"], residual=residual, cfg=cfg, lnm=lnm, eps=eps, cache=cache)
    return out


def _check(z3, w):
    if not z3.is_cuda or z3.dim() != 3 or z3.shape[0] != z3.shape[1] or z3.shape[2] != w["cz"]:
        raise Unsupported("shape %s (want [N, N, %d])" % (tuple(z3.shape), w["cz"]), "shape")
    if z3.dtype not in (torch.bfloat16, torch.float32):
        raise Unsupported("dtype %s (bf16 | fp32-resident z)" % z3.dtype, "dtype")
    if not kernels().has_unit(w["cz"], w["ch"]):
        raise Unsupported("no cubin for (c_z=%d, c_hidden=%d)" % (w["cz"], w["ch"]), "no_cubin")
    form = "f" if z3.dtype == torch.float32 else "b"
    if K.lookup(kernels().arch, w["cz"], w["ch"], form) is None:
        raise Unsupported("no tile-table entry for (c_z=%d, c_hidden=%d, %s z) on %s" % (w["cz"], w["ch"], "fp32" if form == "f" else "bf16", kernels().arch), "no_table")


def trimul_plane(z3, mask2, w, *, outgoing, residual=False, incoming_mode="kt", k1_cfg=None, k3_cfg=None, cache=None, eps=1e-5, lnm=(2, 1)):
    """One pair plane, fast variant: z3 [N, N, cz] contiguous (bf16 | fp32), mask2 None | [N, N] fp32 contiguous."""
    if cache is None:
        cache = _CACHE
    _check(z3, w)
    N = z3.shape[0]; Np = ceil16(N); ch = w["ch"]; cz = w["cz"]
    transpose = (not outgoing) and incoming_mode == "kt"
    kk = kernels()
    stream = kk.stream_handle()
    ab = _buf(cache, ("ab", Np, ch), (2 * ch, Np, Np), torch.bfloat16, z3.device)
    xz = zb = None
    lnm1, lnm3 = (lnm, lnm) if isinstance(lnm, int) else lnm
    form = "f" if z3.dtype == torch.float32 else "b"
    ent = K.lookup(kk.arch, cz, ch, form, N, "outgoing" if outgoing else "incoming", residual)   # deterministic defaults; explicit k1_cfg / k3_cfg override
    f32mode = ent["k3_mode"] if form == "f" else None
    if f32mode == "c":                             # the prologue runs the bf16 kernels: their table entry supplies the K1 default
        entb = K.lookup(kk.arch, cz, ch, "b", N, "outgoing" if outgoing else "incoming", residual)
        k1_cfg = k1_cfg or entb["k1"]; k3b_cfg = k3_cfg or entb["k3"]; k3_cfg = k3_cfg or ent["k3"]
    else:
        k1_cfg = k1_cfg or ent["k1"]; k3_cfg = k3_cfg or ent["k3"]
    if f32mode in ("f", "p") and lnm1 == 2:
        lnm1 = 1                                   # the reference tree is defined on bf16 rows; fp32 rows use the statement's balanced tree on the fp32 values
    if f32mode == "c":                             # one cast pass, then the bf16 prologue on the cast (K3 mode c adds the fp32 residual and writes fp32)
        zb = _buf(cache, ("zcast", N, cz), (N, N, cz), torch.bfloat16, z3.device)
        zb.copy_(z3)
        kk.k1(zb, mask2, w, ab, N=N, Np=Np, cz=cz, ch=ch, cfg=k1_cfg, lnm=lnm1, transpose=transpose, eps=eps, cache=cache, stream=stream)
    else:
        if f32mode == "p":
            xz = _buf(cache, ("xz", N, cz), (N, N, cz), torch.bfloat16, z3.device)   # bf16(LN_in(z)) rows: K3's gate operand (mode p)
        kk.k1(z3, mask2, w, ab, N=N, Np=Np, cz=cz, ch=ch, cfg=k1_cfg, lnm=lnm1, transpose=transpose, xz=xz, eps=eps, cache=cache, stream=stream)
    x = _buf(cache, ("x", Np, ch), (ch, Np, Np), torch.bfloat16, z3.device)
    torch.bmm(ab[:ch], ab[ch:].transpose(1, 2), out=x) if (outgoing or transpose) else torch.bmm(ab[:ch].transpose(1, 2), ab[ch:], out=x)
    if f32mode == "c" and not residual:            # the update alone: returned in the compute dtype (bf16), as any fused bf16 row reading a cast z does
        out = torch.empty((N, N, cz), dtype=torch.bfloat16, device=z3.device)
        kk.k3(x, zb, w, out, N=N, Np=Np, cz=cz, ch=ch, residual=False, cfg=k3b_cfg, lnm=lnm3, eps=eps, cache=cache, stream=stream)
        return out
    if f32mode == "f" and not residual:            # fp32-resident z, update only: K3 mode g reads the fp32 tile and writes the update in bf16
        out = torch.empty((N, N, cz), dtype=torch.bfloat16, device=z3.device)
        kk.k3(x, z3, w, out, N=N, Np=Np, cz=cz, ch=ch, residual=False, cfg=k3_cfg, lnm=lnm3, eps=eps, cache=cache, stream=stream)
        return out
    out = torch.empty((N, N, cz), dtype=z3.dtype, device=z3.device)
    if f32mode == "c":                             # z + update accumulated and stored in fp32
        kk.k3(x, zb, w, out, N=N, Np=Np, cz=cz, ch=ch, residual=residual, cfg=k3_cfg, lnm=lnm3, eps=eps, cache=cache, stream=stream, zres=z3, prenorm=False)
    elif xz is not None:
        kk.k3(x, xz, w, out, N=N, Np=Np, cz=cz, ch=ch, residual=residual, cfg=k3_cfg, lnm=lnm3, eps=eps, cache=cache, stream=stream, zres=z3)
    else:
        kk.k3(x, z3, w, out, N=N, Np=Np, cz=cz, ch=ch, residual=residual, cfg=k3_cfg, lnm=lnm3, eps=eps, cache=cache, stream=stream)
    return out


def trimul_plane_exact(z3, mask2, w, *, outgoing, residual=False, cache=None, eps=1e-5):
    """One pair plane, bitwise variant (bf16 z): reference-order LayerNorms, unpadded planes, the stock einsum contraction."""
    if cache is None:
        cache = _CACHE
    _check(z3, w)
    if z3.dtype != torch.bfloat16:
        raise Unsupported("exact variant serves bf16 z", "exact_dtype")
    N = z3.shape[0]; ch, cz = w["ch"], w["cz"]
    if N % 8 == 0:                         # pitch-N planes straight from K1 (16-byte rows): the stock op's own cuBLAS problem, at its speed
        ab = planes(z3, mask2, w, lnm=2, Np=N, cache=cache, eps=eps)
    else:                                  # N % 8 != 0: 16-padded planes + ONE strided pack copy (bitwise-neutral; K1's 2-byte ragged stores would cost
        ab = planes(z3, mask2, w, lnm=2, cache=cache, eps=eps)[:, :N, :N].contiguous()   # ~4x the copy).  The stock op's GEMM at an unaligned
                                           # pitch is cuBLAS's alignment-1 kernel (5-6x the aligned one): being bitwise there costs what the stock costs.
    a, b = torch.chunk(ab.reshape(2 * ch, 1, N, N), 2, dim=0)
    x = torch.einsum("dbik,dbjk->dbij", a, b) if outgoing else torch.einsum("dbki,dbkj->dbij", a, b)
    x = x.reshape(ch, N, N)
    if N % 8:                              # K3 reads plane rows through 16-byte TMA boxes: ragged N takes one pad-copy of the contraction result
        Np8 = (N + 7) // 8 * 8
        xp = torch.zeros((ch, Np8, Np8), dtype=torch.bfloat16, device=z3.device)
        xp[:, :N, :N].copy_(x)
        x = xp
    return epilogue(x, z3, w, residual=residual, lnm=2 if N % 4 == 0 else 3, cache=cache, eps=eps)


def trimul(z, mask=None, *, direction, weights, residual=False, exact=False, eps=1e-5, cache=None, incoming_mode=None, k1_cfg=None, k3_cfg=None, numerics="ref"):
    """The op (module boundary above).  z [N,N,cz] or [B,N,N,cz] (B > 1: one launch set per plane), bf16 or fp32; mask None | [N,N] | [B|1,N,N].
    Output dtype: bf16 z -> bf16.  fp32-resident z: residual=False returns the UPDATE in bf16 (the compute dtype, at every width: K3 mode g on
    fp32 tiles, or the cast route at c_z = 256); residual=True returns z + update accumulated and stored in fp32.
    numerics: "ref" = the shared statement (csrc/common/tmn_math.cuh; the default tolerance class) | "tx" = trimul_tx 1.2's arithmetic where compiled
    (c_z = c_hidden = 256 bf16; comparison only).  exact=True selects the bitwise variant (reference-order LayerNorms) for bf16 z."""
    outgoing = {"outgoing": True, "incoming": False}[direction]
    if incoming_mode is None:
        incoming_mode = K.INCOMING_MODE            # the tile table's default ("kt": z read transposed by K1, NT contraction)
    if cache is None:
        cache = _CACHE
    w = _packed(weights, cache, z.device)
    if z.dim() == 3 and (mask is None or mask.dim() == 2):          # the common single-plane call: no view churn
        zb = z if z.is_contiguous() else z.contiguous()
        mb = None
        if mask is not None:
            mb = mask if mask.dtype == torch.float32 else mask.to(torch.float32)
            if not mb.is_contiguous():
                mb = mb.contiguous()
        if exact:
            return trimul_plane_exact(zb, mb, w, outgoing=outgoing, residual=residual, cache=cache, eps=eps)
        return trimul_plane(zb, mb, w, outgoing=outgoing, residual=residual, incoming_mode=incoming_mode, k1_cfg=k1_cfg, k3_cfg=k3_cfg, cache=cache, eps=eps, lnm=_LNM[numerics])
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    B, N = z4.shape[0], z4.shape[1]
    m3 = None
    if mask is not None:
        m3 = mask.reshape(-1, N, N)
        if m3.dtype != torch.float32:
            m3 = m3.to(torch.float32)
    outs = []
    for bi in range(B):
        zb = z4[bi].contiguous()
        mb = None if m3 is None else m3[bi if m3.shape[0] == B else 0].contiguous()
        if exact:
            outs.append(trimul_plane_exact(zb, mb, w, outgoing=outgoing, residual=residual, cache=cache, eps=eps))
        else:
            outs.append(trimul_plane(zb, mb, w, outgoing=outgoing, residual=residual, incoming_mode=incoming_mode, k1_cfg=k1_cfg, k3_cfg=k3_cfg, cache=cache, eps=eps, lnm=_LNM[numerics]))
    out = outs[0].unsqueeze(0) if B == 1 else torch.stack(outs, 0)
    return out if z.dim() == 4 else out[0]
