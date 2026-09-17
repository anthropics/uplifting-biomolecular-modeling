"""install.py — the sampler / pairformer attention levers' install functions (instance-level ``forward`` / ``standard_multihead_attention``
on stock modules; stock's Linear projections are used unchanged and consumed as strided views; no stock file is edited).

dit_attn       ENGINEERING lever: the 24 DiffusionTransformer blocks' token attention (q,k,v [S,N,16,48] fp32, pair bias [1,16,N,N]
               step-invariant, broadcast over the S samples in-kernel, gate fused) on ONE fused Triton flash kernel with tf32x3 operands
               (3xTF32: fp32-faithful — the numerics class of the cutlass fp32 kernel stock dispatches to; TOLERANCE only because the
               reduction order differs).  stock = F.scaled_dot_product_attention -> cutlass fmha fp32 + bias/gate glue.
dit_attn_fp16  PRECISION lever on top of dit_attn: the same kernel with fp16 tensor-core operands (q, k, v, P cast in-kernel; bias, logits,
               softmax statistics, accumulation stay fp32) = the TF32-floor error class of the cuBLAS GEMMs around the site; this is the
               lever that buys the speed (5x stock per call vs 1.7x for tf32x3). Requires dit_attn (refuses by name without it); ablating it
               alone leaves the kernel on at tf32x3.
atom_attn      the atom encoder (3) + decoder (3) blocks' local attention (q,k,v [S,N_atom,4,32] fp32, 32x128 windows, bias
               [1,4,n_trunks,32,128]) on one Triton launch, tf32rn operands (= stock's own TF32-cuBLAS class at that site).
               stock = primitives._local_attention (pad + unfold + chunked fp32 math attention).
pf_attn        the Pairformer AttentionPairBias (trunk 48 blocks x N_cycle + confidence 4 blocks; bf16 autocast): fused LN(z)+Linear bias
               producer + the fused core on stock's own bf16 projections (sealed separately; see INTEGRATION.md).
CELLS below hold the measured launch cell per lever per compute capability (dit_attn: per precision state). Composition: install AFTER the
sampler graph / DiT hoist (fpf_clisampler) and sampler_fuse (DIT_FUSE) — the hoist's cached pair bias arrives through the unchanged
AttentionPairBias.forward; DIT_FUSE's F-gate site (Attention._wrap_up) is not reached on the 24 token modules (the gate is fused here: a
SUBSUMED site, its counters re-base), its other kernels are untouched; CUDA-graph capture safe (no syncs, no host-dependent shapes).
Refuse-by-name: any topology / method / shape this file does not know raises RuntimeError naming the lever (no stock fallback).
"""
from __future__ import annotations
import functools, math, os, sys
import torch

CELLS = {                                                   # per lever, per compute capability "major.minor": the measured launch cell
    "dit_attn": {"9.0": {"fp32": dict(opd="tf32x3", bias_tma=True),      # H100, dit_attn alone: 3xTF32 operands (fp32-faithful), 64x64 tiles / 4 warps / 2 stages (apb_config), bias tile via TMA
                         "fp16": dict(opd="fp16", precast=True, bias_tma=True)}},   # H100, dit_attn + dit_attn_fp16: stock's fp32 q/k/v pre-cast to fp16 (3 copy kernels) -> the 16-bit split-D kernel (bitwise = casting in-kernel, 1.3-1.45x faster per call); gate logits read fp32 in the epilogue
    "atom_attn": {"9.0": dict(opd="tf32rn")},               # H100: operands rounded to nearest TF32 (cuBLAS TF32 class), one program per (trunk, sample, head)
    "pf_attn": {"9.0": dict(producer_rows=None, bias_tma=True)},  # H100: fused LN+Linear producer, rows per program = 16384 // c_z (64 @ c_z 256: a [rows, c_z] fp32 tile in registers); bf16 core (operands = stock's own bf16 projections)
}
STATE = {
    "dit_attn": {"installed_on": 0, "calls": 0, "cell": None, "cell_key": None, "named": None, "error": None, "precision": None},
    "dit_attn_fp16": {"on": False, "error": None},
    "atom_attn": {"installed_on": 0, "calls": 0, "cell": None, "cell_key": None, "named": None, "error": None},
    "pf_attn": {"installed_on": 0, "calls": 0, "cell": None, "cell_key": None, "named": None, "error": None, "skipped_has_s": 0},
}


def _cell(lever: str, sub: str | None = None) -> dict:
    """The lever's cell for this card (dit_attn: per precision state `sub`); an unmeasured card engages the sm_90 cell and is NAMED."""
    cc = "%d.%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "cpu"
    table = CELLS[lever]
    key = cc if cc in table else "9.0"
    STATE[lever]["cell_key"] = key
    if key != cc:
        STATE[lever]["named"] = f"untested arch sm_{cc.replace('.', '')}: engaged with the sm_90 cell"
    row = table[key]
    return dict(row[sub] if sub is not None else row)


def _diffusion_module(model):
    from protenix.model.modules import diffusion as Dm
    dms = [m for m in model.modules() if isinstance(m, Dm.DiffusionModule)]
    if len(dms) != 1:
        raise RuntimeError(f"expected one DiffusionModule in the model, found {len(dms)}")
    return dms[0]


# ----------------------------------------------------------------------------------------------------------------- dit_attn
def _make_dit_forward(att, cell):
    from .apb_triton import apb_views, precast16
    H = att.num_heads
    opd = cell["opd"]; bias_tma = cell.get("bias_tma", True); precast = bool(cell.get("precast", False))
    st = STATE["dit_attn"]

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        if n_queries or trunked_attn_bias is not None:
            raise RuntimeError("dit_attn: a local-attention call reached the DiT token attention lever (n_queries / trunked_attn_bias set)")
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        C = q.shape[-1]; D = C // H
        lead = q.shape[:-2]; N = q.shape[-2]
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        S = q4.shape[0]
        if k4.shape[0] != S:
            raise RuntimeError(f"dit_attn: q/kv sample dims differ ({tuple(q_x.shape)} vs {tuple(kv_x.shape)})")
        if attn_bias is None:
            raise RuntimeError("dit_attn: attn_bias=None at the DiT token attention (the model always passes the pair bias here)")
        b = attn_bias
        while b.dim() > 3 and b.shape[0] == 1:
            b = b[0]
        if b.dim() == 4 and b.shape[0] == S and b.shape[1] == 1 and S != H:   # [S,1,N,N]: a per-sample mask-only bias — not this site's contract
            raise RuntimeError(f"dit_attn: per-sample attn_bias {tuple(attn_bias.shape)} is not supported (expected the pair bias [1,{H},N,N] shared by the samples)")
        if b.dim() != 3 or b.shape[0] != H or b.shape[-1] < N or b.shape[-2] < N:
            raise RuntimeError(f"dit_attn: unexpected attn_bias shape {tuple(attn_bias.shape)} for q {tuple(q_x.shape)} (expected [1,{H},{N},{N}])")
        if b.shape[-1] != N or b.shape[-2] != N:
            b = b[:, :N, :N]
        if b.stride(-1) != 1:
            b = b.contiguous()
        cfg = None if bias_tma else dict(BIAS_TMA=False)
        if precast and q4.dtype == torch.float32:
            q4, k4, v4 = precast16(q4, k4, v4, torch.float16 if opd == "fp16" else torch.bfloat16)
            o = apb_views(q4, k4, v4, b, g4, scale=1.0 / math.sqrt(D), out_dtype=torch.float32, cfg=cfg)
        else:
            o = apb_views(q4, k4, v4, b, g4, scale=1.0 / math.sqrt(D), out_dtype=q.dtype, opd=(opd if q.dtype == torch.float32 else None), cfg=cfg)
        st["calls"] += 1
        return att.linear_o(o.view(*lead, N, C))

    forward.__wrapped__ = type(att).forward; forward._fpf_apb = "dit_attn"
    return forward


def install_dit_attn(model, fp16: bool = False) -> dict:
    """Put the fused kernel on the 24 DiffusionTransformer token-attention modules of the model's DiffusionModule.
    fp16=False: lever dit_attn alone (tf32x3 operands, fp32-faithful).  fp16=True: dit_attn + the precision lever dit_attn_fp16 (fp16
    operands). The kit calls this ONCE with fp16 = (PTX_DIT_ATTN_FP16 == "1"); PTX_DIT_ATTN_FP16=1 without PTX_DIT_ATTN=1 must be refused
    by the caller by name (install_dit_attn_fp16_requires_dit_attn) — this function is the only install point of both levers."""
    st = STATE["dit_attn"]
    try:
        from protenix.model.modules import primitives as P
        from . import apb_triton  # noqa: F401  (imports triton; TensorDescriptor optional)
        st["precision"] = "fp16" if fp16 else "fp32"
        STATE["dit_attn_fp16"]["on"] = bool(fp16)
        cell = _cell("dit_attn", st["precision"])
        if cell.get("bias_tma", True) and not apb_triton._HAS_TMA:
            cell["bias_tma"] = False
            st["named"] = ((st["named"] + "; ") if st["named"] else "") + "triton without host TMA descriptors: bias tile via plain loads"
        dm = _diffusion_module(model)
        mods = [blk.attention_pair_bias.attention for blk in dm.diffusion_transformer.blocks]
        if len(mods) != 24 or any(not isinstance(m, P.Attention) for m in mods):
            raise RuntimeError(f"expected 24 primitives.Attention token modules under diffusion_transformer.blocks, found {len(mods)}")
        for att in mods:
            if att.num_heads != 16 or att.c_hidden != 48:                       # primitives.Attention: c_hidden IS the per-head width (linear_q: c_hidden * num_heads)
                raise RuntimeError(f"token attention geometry heads={att.num_heads} c_hidden={att.c_hidden} (cells measured for 16 x 48)")
            att.forward = _make_dit_forward(att, cell)
        st["installed_on"] = len(mods); st["cell"] = cell
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise RuntimeError(f"dit_attn: {e}") from e


# ----------------------------------------------------------------------------------------------------------------- atom_attn
def _make_atom_forward(att, cell):
    from .atom_triton import atom_apb
    H = att.num_heads
    opd = cell["opd"]
    st = STATE["atom_attn"]

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        if not (n_queries and n_keys) or trunked_attn_bias is None or attn_bias is not None:
            raise RuntimeError(f"atom_attn: unexpected call (n_queries={n_queries}, n_keys={n_keys}, trunked_bias={'set' if trunked_attn_bias is not None else None}, attn_bias={'set' if attn_bias is not None else None})")
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        C = q.shape[-1]; D = C // H
        lead = q.shape[:-2]; N = q.shape[-2]
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        if k4.shape[0] != q4.shape[0]:
            raise RuntimeError(f"atom_attn: q/kv sample dims differ ({tuple(q_x.shape)} vs {tuple(kv_x.shape)})")
        b = trunked_attn_bias
        while b.dim() > 4:
            if b.shape[0] != 1:
                raise RuntimeError(f"atom_attn: per-sample local pair bias {tuple(trunked_attn_bias.shape)} (the model's is sample-invariant [1,{H},n_trunks,{n_queries},{n_keys}])")
            b = b[0]
        if b.stride(-1) != 1:
            b = b.contiguous()
        o = atom_apb(q4, k4, v4, b, g4, n_queries=n_queries, n_keys=n_keys, scale=1.0 / math.sqrt(D), out_dtype=q.dtype, opd=(opd if q.dtype == torch.float32 else None))
        st["calls"] += 1
        return att.linear_o(o.view(*lead, N, C))

    forward.__wrapped__ = type(att).forward; forward._fpf_apb = "atom_attn"
    return forward


def install_atom_attn(model) -> dict:
    """Put the fused local-window kernel on the 6 atom-transformer attention modules of the DiffusionModule's atom encoder + decoder."""
    st = STATE["atom_attn"]
    try:
        from protenix.model.modules import primitives as P, transformer as T
        from . import atom_triton  # noqa: F401
        cell = _cell("atom_attn")
        dm = _diffusion_module(model)
        mods = []
        for part in (dm.atom_attention_encoder, dm.atom_attention_decoder):
            for m in part.modules():
                if isinstance(m, T.AtomTransformer):
                    mods += [blk.attention_pair_bias.attention for blk in m.diffusion_transformer.blocks]
        if len(mods) != 6 or any(not isinstance(m, P.Attention) for m in mods):
            raise RuntimeError(f"expected 6 primitives.Attention modules under the diffusion module's atom encoder/decoder AtomTransformers, found {len(mods)}")
        for att in mods:
            meth = getattr(att, "local_attention_method", None)
            if meth != "local_cross_attention":
                raise RuntimeError(f"local_attention_method={meth!r} (the kernel implements 'local_cross_attention')")
            if att.num_heads != 4 or att.c_hidden != 32:
                raise RuntimeError(f"atom attention geometry heads={att.num_heads} c_hidden={att.c_hidden} (cells measured for 4 x 32)")
            att.forward = _make_atom_forward(att, cell)
        st["installed_on"] = len(mods); st["cell"] = cell
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise RuntimeError(f"atom_attn: {e}") from e


def _make_pf_smha(apb, cell):
    """AttentionPairBias.standard_multihead_attention replacement (has_s=False self-attention in the Pairformer stacks): fused LN(z)+Linear
    bias producer (pf_bias) + the fused bias-attention core on stock's own bf16 q/k/v/g projections (gate epilogue) + stock's linear_o."""
    from .pf_triton import pf_bias
    from .apb_triton import apb_views
    st = STATE["pf_attn"]; att = apb.attention
    R = cell.get("producer_rows", 128)

    def smha(q, kv, z, inplace_safe=False, enable_efficient_fusion=False):
        if enable_efficient_fusion or kv is not q:
            raise RuntimeError(f"pf_attn: unexpected call (enable_efficient_fusion={enable_efficient_fusion}, cross-attention={kv is not q})")
        lead = q.shape[:-2]; N = q.shape[-2]
        zz = z.reshape(-1, *z.shape[-3:])
        if zz.shape[0] != 1 or zz.shape[1] != N or zz.shape[2] != N:
            raise RuntimeError(f"pf_attn: pair tensor {tuple(z.shape)} vs single {tuple(q.shape)} (batched pair biases are not wired)")
        zz = zz[0]
        if zz.stride(2) != 1 or zz.stride(0) != N * zz.stride(1):
            zz = zz.contiguous()
        ln = apb.layernorm_z
        bias = pf_bias(zz, ln.weight, getattr(ln, "bias", None), apb.linear_nobias_z.weight, getattr(ln, "eps", 1e-5), R=R, out_dtype=torch.float32)   # [H, N, N] bf16-valued logits in an fp32 container (pitch-8 view): what stock's .float() hands the core; measured = or faster than a bf16 container
        H, D = att.num_heads, att.c_hidden
        a2 = q.reshape(-1, N, q.shape[-1]); S = a2.shape[0]
        qh = att.linear_q(a2).view(S, N, H, D); kh = att.linear_k(a2).view(S, N, H, D); vh = att.linear_v(a2).view(S, N, H, D)
        gh = att.linear_g(a2).view(S, N, H, D) if att.gating else None
        o = apb_views(qh, kh, vh, bias, gh, scale=1.0 / math.sqrt(D), out_dtype=qh.dtype, opd=("fp16" if qh.dtype == torch.float32 else None))
        st["calls"] += 1
        return att.linear_o(o.view(*lead, N, H * D))

    smha.__wrapped__ = type(apb).standard_multihead_attention; smha._fpf_apb = "pf_attn"
    return smha


def install_pf_attn(model) -> dict:
    """Put the fused producer + core on every Pairformer AttentionPairBias (has_s=False) outside the DiffusionModule: the trunk PairformerStack
    (48 blocks) and the ConfidenceHead pairformer (4 blocks); MSA-module / template pair stacks have no AttentionPairBias (c_s = 0)."""
    st = STATE["pf_attn"]
    try:
        from protenix.model.modules import transformer as T, diffusion as Dm
        from . import pf_triton  # noqa: F401
        cell = _cell("pf_attn")
        dm_ids = set()
        for m in model.modules():
            if isinstance(m, Dm.DiffusionModule):
                dm_ids |= {id(x) for x in m.modules()}
        n = 0
        for name, m in model.named_modules():
            if isinstance(m, T.AttentionPairBias) and id(m) not in dm_ids:
                if m.has_s or getattr(m, "cross_attention_mode", False):
                    st["skipped_has_s"] += 1
                    continue
                att = m.attention
                if att.num_heads != 16 or att.c_hidden not in (16, 24, 32, 48, 64) or not att.gating:
                    raise RuntimeError(f"{name}: heads={att.num_heads} c_hidden={att.c_hidden} gating={att.gating} (cells measured for 16 gated heads)")
                c_z = m.linear_nobias_z.in_features
                if c_z & (c_z - 1) or c_z < 16:
                    raise RuntimeError(f"{name}: c_z={c_z} (the producer needs a power-of-two c_z >= 16)")
                m.standard_multihead_attention = _make_pf_smha(m, cell)
                n += 1
        if n == 0:
            raise RuntimeError("no Pairformer AttentionPairBias (has_s=False) modules found outside the DiffusionModule")
        st["installed_on"] = n; st["cell"] = cell
        return dict(st)
    except Exception as e:
        st["error"] = repr(e)
        raise RuntimeError(f"pf_attn: {e}") from e


def report() -> dict:
    """Counters for the LEVER / EXIT lines: Python-level calls (under the sampler graph these are the eager warm-up + capture calls; the
    replays run the captured kernels), modules installed, the cell used and any NAMED condition."""
    return {k: dict(v) for k, v in STATE.items()}
