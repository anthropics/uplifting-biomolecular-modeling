# Copyright 2026 Anthropic, PBC (integrator wiring of the PAD8-EXACT finding). Apache-2.0.
"""fpf_pad8exact — 'PAD8-EXACT' provider for the FlashPairformer padded core: the STOCK cuEquivariance triangle_attention op called on ceil8(N)-padded q/k/v VIEWS
(shared zero-filled per-P buffers owned by the core) with bias columns [N, P) pre-filled -1e9 and mask=None.  Probe-track claim (H100, cuEq 0.8.0 + 0.11.1, 356..1493,
both nodes): the same sm80 kernel runs and the valid-region output is BITWISE == the stock (unpadded, mask=None) call, x1.13-1.17 faster per call at 546-1493.
Label: EXACT-candidate until the CLI DET all-files test record exists (then EXACT on cc 8.x/9.x).  cc 10.x is refused by the levers (aligned unmasked calls reach cudnn_sm100 there).
Contract (v0.4 provider interface): attn(q5,k5,v5,bias4,mask=None,scale=,kv_len=N,q_len=rows) -> o [1?, rows|P, H, P, D]; rows >= q_len / cols >= kv_len are don't-care (the epilogue crops by strides)."""
import torch
__version__ = "0.1.0"
PROVIDER = {"name": "pad8exact", "label": "EXACT-candidate", "kv_pad": 8, "bias_pad_fill": -1e9, "wants_kv_len": True, "zero_kv_pad": True}
STATS = {"calls": 0, "padded_calls": 0, "aligned_calls": 0, "refused": 0}

class Refused(Exception):
    pass

def attn(q, k, v, bias, mask=None, scale=None, kv_len=None, q_len=None, **kw):
    import protenix.model.triangular.layers as TL
    STATS["calls"] += 1
    if mask is not None:
        STATS["refused"] += 1; raise Refused("dense_mask")
    q4, k4, v4 = (t[0] if t.dim() == 5 and t.shape[0] == 1 else t for t in (q, k, v))
    b4 = bias if bias.dim() == 4 else bias.unsqueeze(0)
    if kv_len is not None and int(kv_len) < k4.shape[-2]: STATS["padded_calls"] += 1
    else: STATS["aligned_calls"] += 1
    o = TL.cuequivariance_triangular_attn(q4, k4, v4, (b4 if b4.dtype == torch.float32 else b4.float()), None, scale)      # the stock op, stock arguments (mask=None as the NOMASK lever does)
    return o[0] if isinstance(o, (tuple, list)) else o

def attn_unpadded_marker(*a, **k):          # registered in _BLK_ATT['fn'] only so the core evaluates the provider branch; never called (the core maps it to the stock cuEq path)
    raise Refused("marker")
attn_unpadded_marker._fpf_pad8_marker = True
