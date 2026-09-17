"""dit_fuse_patch.py — install the F-levers (dit_fuse) into a Protenix v2 model at runtime (no source edits). Hooks:
  * every AdaptiveLayerNorm inside diffusion_module (token blocks: apb.layernorm_a, ctb.adaln; atom blocks: + apb.layernorm_kv)  -> F-ada
  * every token-block Attention (global path, gating=True): _wrap_up -> F-gate                                             -> F-gate
  * every AttentionPairBias with has_s (output gate) + the DiffusionTransformerBlock residual                               -> F-res
  * every ConditionedTransitionBlock: silu(a1(a))*a2(a) and sigmoid(linear_s(s))*linear_b(b) (+ block residual)              -> F-swiglu + F-res
Composes with dit_hoist (the hoist caches the s-side producers; we only replace the elementwise consumers: we call the SAME sub-module forwards, so
hoist hooks on linear_s/linear_nobias_s/linear_a_last/layernorm_s still fire) and with dit_levers (not both on the same module: if dit_levers replaced
apb/ctb forward we skip those modules and count them).
Env: DIT_FUSE (comma list of ada,gate,res,swiglu; default all), DIT_FUSE_SCOPE=token|atom|all (default all).
"""
from __future__ import annotations
import os, sys, types
import torch
import torch.nn.functional as F
import dit_fuse as DF

REPORT = {"installed": False, "n_ada": 0, "n_gate": 0, "n_res_apb": 0, "n_ctb": 0, "n_blocks": 0, "skipped": [], "levers": None, "scope": None}


def _dm(model):
    m = getattr(model, "diffusion_module", None)
    if m is None and hasattr(model, "module"): m = getattr(model.module, "diffusion_module", None)
    assert m is not None, "model has no diffusion_module"
    return m


def install(model, levers: str | None = None, scope: str | None = None):
    from protenix.model.modules.primitives import AdaptiveLayerNorm, Attention
    from protenix.model.modules.transformer import AttentionPairBias, ConditionedTransitionBlock, DiffusionTransformerBlock
    levers = set((levers or os.environ.get("DIT_FUSE", "ada,gate,res,swiglu")).split(","))
    scope = scope or os.environ.get("DIT_FUSE_SCOPE", "all")
    REPORT["levers"] = sorted(levers); REPORT["scope"] = scope
    dm = _dm(model)
    tok_blocks = list(dm.diffusion_transformer.blocks)
    atom_blocks = list(dm.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks) + list(dm.atom_attention_decoder.atom_transformer.diffusion_transformer.blocks)
    blocks = (tok_blocks if scope in ("token", "all") else []) + (atom_blocks if scope in ("atom", "all") else [])
    for blk in blocks:
        assert isinstance(blk, DiffusionTransformerBlock), type(blk)
        apb, ctb = blk.attention_pair_bias, blk.conditioned_transition_block
        levered = getattr(apb, "_dit_levers", False) or ("forward" in apb.__dict__ and "dit_levers" in repr(apb.__dict__.get("forward")))
        # ---- F-ada on the AdaLNs (instance-level forward override; calls the module's own LN / linear sub-modules => hoist hooks still apply)
        if "ada" in levers:
            adas = [apb.layernorm_a, ctb.adaln] + ([apb.layernorm_kv] if getattr(apb, "cross_attention_mode", False) and apb.has_s else [])
            for ada in adas:
                if not isinstance(ada, AdaptiveLayerNorm) or "forward" in ada.__dict__ and getattr(ada, "_dit_fuse", False):
                    continue
                prev = ada.__dict__.get("forward", None)     # an instance-level forward is already set (dit_hoist's ada_fwd on atom blocks):
                if prev is not None:
                    # dit_hoist replaced this AdaLN (atom blocks: g and b cached). Its tail is g*a + b (2 kernels) — leave it (already minimal) and count.
                    REPORT["skipped_hoisted_ada"] = REPORT.get("skipped_hoisted_ada", 0) + 1; continue
                def ada_fwd(a, s, _ada=ada):
                    a_n = _ada.layernorm_a(a)
                    s_n = _ada.layernorm_s(s)
                    x1 = _ada.linear_s(s_n); x2 = _ada.linear_nobias_s(s_n)
                    if a_n.dtype != torch.float32 or x1.shape != a_n.shape:
                        DF.STATS["fallback"] += 1
                        return torch.sigmoid(x1) * a_n + x2
                    out = DF.ada_tail(a_n.contiguous(), x1.contiguous(), x2.contiguous())
                    return out
                ada.forward = ada_fwd; ada._dit_fuse = True; REPORT["n_ada"] += 1
        # ---- F-gate on the token attention (global attention with gating): replace _wrap_up
        att = apb.attention
        if "gate" in levers and isinstance(att, Attention) and att.linear_g is not None and "_wrap_up" not in att.__dict__:
            def wrap_up(o, q_x, _att=att):
                # stock: g = sigmoid(linear_g(q_x)).view(.., H, C); o = o(transposed to [..,Q,H,C]) * g; flatten; linear_o
                # Attention.forward calls _wrap_up AFTER o = o.transpose(-2,-3) (a view: [..,Q,H,C] with strides of [..,H,Q,C])
                gl = _att.linear_g(q_x)                                   # [.., Q, H*C]
                ot = o.transpose(-2, -3)                                  # [.., H, Q, C] view
                if o.dtype == torch.float32 and o.is_contiguous() and gl.is_contiguous() and o.dim() >= 3:
                    # mem-efficient SDPA returns query-major storage: o [.., Q, H, C] is contiguous -> stock = sigmoid | mul on congruent contiguous
                    # tensors (flatten is a free view) -> one kernel: out = o_flat * sigmoid(gl)  (a*b == b*a bitwise)
                    og = DF.res_gate(gl.reshape(-1, gl.shape[-1]), o.reshape(-1, gl.shape[-1]), None, _kind="gate").reshape(gl.shape)
                elif o.dtype == torch.float32 and ot.is_contiguous() and o.dim() >= 3:
                    og = DF.attn_gate(ot, gl.contiguous())                # head-major storage: gather kernel (3 -> 1)
                else:
                    DF.STATS["fallback"] += 1
                    g = _att.sigmoid(gl).view(gl.shape[:-1] + (_att.num_heads, -1)); o2 = (o * g); o2 = o2.reshape(o2.shape[:-2] + (-1,))
                    return _att.linear_o(o2)
                return _att.linear_o(og)
            att._wrap_up = wrap_up; REPORT["n_gate"] += 1
        # ---- F-res: APB output gate fused with the block residual; F-swiglu + F-res in the CTB fused with the second block residual.
        apb_inst_patched = "forward" in apb.__dict__
        ctb_inst_patched = "forward" in ctb.__dict__
        if apb_inst_patched and ctb_inst_patched:
            REPORT["levered_blocks"] = REPORT.get("levered_blocks", 0) + 1      # dit_levers (G) owns this token block; it calls dit_fuse epilogues itself
            REPORT["n_blocks"] += 1
            continue
        if ("res" in levers or "swiglu" in levers) and "forward" not in blk.__dict__:
            o_apb_fwd = apb.forward
            def blk_forward(a, s, z, n_queries=None, n_keys=None, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False, _blk=blk, _apb=apb, _ctb=ctb):
                # ----- attention half: replicate AttentionPairBias.forward up to (not incl.) the output gate, then F-res(gate, a_att, residual a)
                if "res" in levers and not apb_inst_patched and _apb.has_s:
                    x = _apb.layernorm_a(a=a, s=s) if _apb.has_s else _apb.layernorm_a(a)
                    # stock: kv = layernorm_kv(a=a_NORMED, s) — the AdaLN'd a, not the block input (cross_attention_mode atom blocks)
                    kv = (_apb.layernorm_kv(a=x, s=s) if _apb.has_s else _apb.layernorm_kv(x)) if _apb.cross_attention_mode else None
                    if n_queries and n_keys:
                        x = _apb.local_multihead_attention(x, kv if _apb.cross_attention_mode else x, z, n_queries, n_keys, inplace_safe=inplace_safe, chunk_size=chunk_size)
                    else:
                        x = _apb.standard_multihead_attention(x, kv if _apb.cross_attention_mode else x, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
                    gl = _apb.linear_a_last(s)                            # hoist caches this for atom blocks (s == c_l); token blocks: per step
                    if gl.shape != x.shape:                               # broadcasting (N_sample dim): stock broadcasts inside the mul; expand so the fused kernel sees congruent shapes
                        gl = gl.expand_as(x)
                    if x.dtype == torch.float32 and a.shape == x.shape:
                        attn_out = DF.res_gate(gl.contiguous(), x.contiguous(), a.contiguous())     # sigmoid(gl)*x + a
                    else:
                        DF.STATS["fallback"] += 1; attn_out = torch.sigmoid(gl) * x + a
                else:
                    attn_out = _apb(a=a, s=s, z=z, n_queries=n_queries, n_keys=n_keys, inplace_safe=inplace_safe, chunk_size=chunk_size, enable_efficient_fusion=enable_efficient_fusion)
                    attn_out = attn_out + a
                # ----- transition half
                if not ctb_inst_patched and ("swiglu" in levers or "res" in levers):
                    an = _ctb.adaln(attn_out, s)
                    x1 = _ctb.linear_nobias_a1(an); x2 = _ctb.linear_nobias_a2(an)
                    if "swiglu" in levers and x1.dtype == torch.float32:
                        b = DF.swiglu(x1.contiguous(), x2.contiguous())
                    else:
                        b = F.silu(x1) * x2
                    hb = _ctb.linear_nobias_b(b)
                    gl2 = _ctb.linear_s(s)
                    if gl2.shape != hb.shape: gl2 = gl2.expand_as(hb)
                    if "res" in levers and hb.dtype == torch.float32:
                        out_a = DF.res_gate(gl2.contiguous(), hb.contiguous(), attn_out.contiguous())   # sigmoid(linear_s(s))*linear_b(b) + attn_out
                    else:
                        out_a = torch.sigmoid(gl2) * hb + attn_out
                else:
                    out_a = _ctb(a=attn_out, s=s) + attn_out
                return out_a, s, z
            blk.forward = blk_forward; REPORT["n_res_apb"] += int("res" in levers and not apb_inst_patched); REPORT["n_ctb"] += int(not ctb_inst_patched)
        REPORT["n_blocks"] += 1
    REPORT["installed"] = True
    REPORT["cfg"] = DF.STATS["cfg"]
    os.environ["DIT_FUSE_ACTIVE"] = ",".join(sorted(levers))     # read by dit_levers (G) to use the fused epilogues inside its own block forwards
    import atexit
    atexit.register(lambda: print(f"[dit_fuse] EXIT STATS calls={DF.STATS['calls']} fallback={DF.STATS['fallback']}", flush=True))
    print(f"[dit_fuse] installed: {REPORT}", file=sys.stderr, flush=True)
    return REPORT


def summary():
    return {"report": REPORT, "stats": {k: (dict(v) if isinstance(v, dict) else v) for k, v in DF.STATS.items()}}
