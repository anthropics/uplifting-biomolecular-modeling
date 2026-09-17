"""Deterministic test inputs (numpy only, bit-defined on any host) and the recorded output digests (vectors.json) of each row.
Inputs are drawn with numpy's frozen legacy generator (RandomState) and rounded to bfloat16 here, so no framework decides a bit.
A row reproduces a case when sha256(output bytes) equals the digest recorded for (case, row); the digests were produced by the
torch-launched kernels themselves (K2B through Triton's launcher, the CUDA kernel through its torch binding) on the build stack, so
equality is equality with those launches, bit for bit (one K2B digest per cubin arch: the compiler emits different tensor-core code for sm_90 and sm_80)."""
import hashlib
import json
import os
from typing import Dict

from . import PKG_DIR

# name -> (B, N, H, S, D, dtype, mask kind, bias kind, seed)
CASES: Dict[str, tuple] = {
    "bf16_d32_s200_ragged":   (1, 200, 4, 200, 32, "bf16", "ragged", "bf16exact", 1),     # S % 16 != 0, S < 256: BIAS16 off, 'any' cubins; ragged rows + one fully-masked + one hole row
    "bf16_d32_s272_nomask":   (1, 272, 2, 272, 32, "bf16", "none", "bf16exact", 2),       # S % 16 == 0, S >= 256: BIAS16 on (lossless), 's16' cubins
    "bf16_d32_s300_ragged":   (1, 300, 2, 300, 32, "bf16", "ragged", "f32", 3),           # BIAS16 on but the fp32 bias is not bf16-exact (flags 0 path), 'any'
    "bf16_d32_s272_pad":      (1, 272, 4, 272, 32, "bf16", "pad", "bf16exact", 4),        # one padding mask shared by every row (the common model case)
    "bf16_d16_s200_ragged":   (1, 200, 4, 200, 16, "bf16", "ragged", "bf16exact", 5),     # template-stack head_dim
    "bf16_d32_b2_s150_ragged": (2, 150, 2, 150, 32, "bf16", "ragged", "bf16exact", 6),    # B = 2
    "fp32_d32_s200_ragged":   (1, 200, 4, 200, 32, "fp32", "ragged", "f32", 7),
    "fp32_d16_s144_pad":      (1, 144, 4, 144, 16, "fp32", "pad", "f32", 8),
    "bf16_d32_n96_s512_pad":  (1, 96, 4, 512, 32, "bf16", "pad", "bf16exact", 9),      # S >= 512: the fleet kernel's band (row triattn_native); N != S
    "bf16_d32_n64_s640_ragged": (1, 64, 2, 640, 32, "bf16", "ragged", "f32", 10),    # ragged rows + fully-masked + hole rows at S 640, fp32 (inexact) bias
    "bf16_d32_n48_s768_nomask": (1, 48, 4, 768, 32, "bf16", "none", "bf16exact", 11),
    "bf16_d32_n32_s1344_pad": (1, 32, 4, 1344, 32, "bf16", "pad", "bf16exact", 12),      # S > 1280: the sm_80 member's large-S geometry (128-query tiles); the sm_90a rows' large-S tiles too
    "bf16_d16_n40_s512_pad":  (1, 40, 4, 512, 16, "bf16", "pad", "bf16exact", 13),       # head_dim 16 at S 512 (template-stack shape class; k2b_aot + cuda_80 only)
}
PER_ARCH_OPTIONAL = ("bf16_d32_n96_s512_pad", "bf16_d32_n64_s640_ragged", "bf16_d32_n48_s768_nomask", "bf16_d32_n32_s1344_pad", "bf16_d16_n40_s512_pad")   # cases whose per-arch K2B digest may be absent (recorded where a torch box of that arch ran them)


def f32_to_bf16_bits(x):
    import numpy as np
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = b + 0x7FFF + ((b >> 16) & 1)                       # round to nearest even (inputs are finite)
    return (b >> 16).astype(np.uint16)


def bf16_bits_to_f32(u):
    import numpy as np
    return (np.ascontiguousarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def inputs(name: str) -> dict:
    """{'q','k','v': uint16 bf16 bits or float32 [B,N,H,S,D]; 'bias': float32 [B,H,S,S]; 'mask': uint8 [B,N,S] or None; 'meta': {...}}"""
    import numpy as np
    B, N, H, S, D, dtype, mkind, bkind, seed = CASES[name]
    rs = np.random.RandomState(seed)

    def draw(shape, sd=1.0):
        return (rs.standard_normal(size=shape) * sd).astype(np.float32)

    q, k, v = draw((B, N, H, S, D)), draw((B, N, H, S, D)), draw((B, N, H, S, D))
    bias = draw((B, H, S, S), 2.0)
    if bkind == "bf16exact":
        bias = bf16_bits_to_f32(f32_to_bf16_bits(bias))
    mask = None
    if mkind != "none":
        mask = np.ones((B, N, S), np.uint8)
        for b in range(B):
            if mkind == "pad":
                keep = S - 7 - b
                mask[b, :, keep:] = 0
            else:
                for i in range(N):
                    keep = S - ((i * 7 + b) % 23)
                    mask[b, i, keep:] = 0
                mask[b, 5 % N, :] = 0                          # a fully-masked row (uniform attention = mean of v)
                mask[b, 9 % N, 1:S:3] = 0                      # interior holes
                mask[b, 11 % N, :] = 0; mask[b, 11 % N, 0] = 1  # a single attended key
    out = {"bias": bias, "mask": mask, "meta": {"B": B, "N": N, "H": H, "S": S, "D": D, "dtype": dtype, "mask_kind": mkind, "bias_kind": bkind, "seed": seed}}
    if dtype == "bf16":
        out["q"], out["k"], out["v"] = f32_to_bf16_bits(q), f32_to_bf16_bits(k), f32_to_bf16_bits(v)
    else:
        out["q"], out["k"], out["v"] = q, k, v
    return out


def digest(arr) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def input_digests(name: str) -> Dict[str, str]:
    d = inputs(name)
    out = {k: digest(d[k]) for k in ("q", "k", "v", "bias")}
    out["mask"] = digest(d["mask"]) if d["mask"] is not None else "none"
    return out


def recorded() -> dict:
    p = os.path.join(PKG_DIR, "vectors.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
