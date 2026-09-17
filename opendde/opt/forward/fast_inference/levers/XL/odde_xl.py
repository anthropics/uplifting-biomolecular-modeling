# odde_xl.py -- OPENDDE_XL_ADDON: single-card XL memory levers for OpenDDE 1.1.1 (pure monkeypatch; no site-packages edit).
#
# Lever (env ODDE_XL=tri_ln | all):
#   tri_ln     : TriangleAttention prologue -- the full LayerNorm'd copy of the pair tensor is freed BEFORE the chunked attention runs and
#                LN is recomputed per row-chunk (same chunk_size, same q/k/v/o GEMM shapes and the SAME full-M bias GEMM as stock).  Removes one
#                pair-tensor-equivalent from the tri-attention peak (matters most in the structural-token refiner at ~2N tokens, c_z=384).
#                Pattern: the chunked tri-attention prologue (row-chunked pair ops), applied to the stock module.
#   (allocator) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is an ENV setting, not code.
# The lever is arithmetic-preserving by construction (same kernels, same GEMM shapes, same reduction orders): byte-identical to the stock CLI
# under the deterministic recipe.  ODDE_XL_CROSSCHECK=1 asserts LN-recompute == stock LN in-process (cross-check allocates; use it on small items).
import os, sys, json
import torch

__version__ = "0.1.2"
# RAN-OR-REFUSE counters (every key present from import; a lever that installed but never ran reads 0):
#   tri_ln_calls / tri_ln_chunked_calls  TriangleAttention.forward through the lever / of those, the chunked (LN-freeing) regime
_STATS = {"tri_ln_calls": 0, "tri_ln_chunked_calls": 0, "crosscheck_fail": 0, "crosscheck_ok": 0}
KNOWN = {"tri_ln"}


def requested():
    v = os.environ.get("ODDE_XL", "").strip()
    if not v or v in ("0", "off"):
        return set()
    if v in ("1", "all"):
        return {"tri_ln"}
    return {x.strip() for x in v.split(",") if x.strip()}


def log(msg):
    print(f"[odde_xl] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------------------- tri_ln
def _install_tri_ln():
    from opendde.model.triangular import triangular as T
    from opendde.model.utils import chunk_layer, permute_final_dims
    from functools import partial
    crosscheck = os.environ.get("ODDE_XL_CROSSCHECK", "0") == "1"
    stock_forward = T.TriangleAttention.forward

    def forward_xl(self, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
        _STATS["tri_ln_calls"] += 1
        if chunk_size is None:
            # un-chunked regime (stock: <= 1024 tokens): nothing to save without changing GEMM shapes -> stock body verbatim
            return stock_forward(self, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        _STATS["tri_ln_chunked_calls"] += 1
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)
        # (1) bias exactly as stock: LN over the full tensor, full-M Linear(c_in -> H); keep only the small bias, free the LN copy
        x_ln = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = permute_final_dims(self.linear(x_ln), (2, 0, 1))   # [*, H, I, J] view of the small [*, I, J, H] Linear output (as stock)
        triangle_bias = triangle_bias.unsqueeze(-4)                         # [*, 1, H, I, J]
        keep = x_ln[..., :2, :, :].clone() if crosscheck else None
        del x_ln
        biases = [mask_bias, triangle_bias]
        ln = self.layer_norm
        mha = self.mha

        def mha_ln(q_x, kv_x, biases, triangle_attention="torch"):
            xc = ln(q_x)                       # q_x and kv_x are the same raw row-chunk (chunk_layer slices both from x)
            return mha(q_x=xc, kv_x=xc, biases=biases, triangle_attention=triangle_attention)

        if crosscheck:
            chk = ln(x[..., :2, :, :])
            ok = torch.equal(chk, keep)
            _STATS["crosscheck_ok" if ok else "crosscheck_fail"] += 1
            if not ok:
                log(f"CROSSCHECK FAIL tri_ln: per-chunk LN != full LN (max abs diff {(chk - keep).abs().max().item():.3e})")
            del chk, keep
        # (2) chunked attention exactly as stock's _chunk: same chunk_size over the same batch dims -> same per-chunk GEMM shapes
        out = chunk_layer(partial(mha_ln, triangle_attention=triangle_attention),
                          {"q_x": x, "kv_x": x, "biases": biases}, chunk_size=chunk_size, no_batch_dims=len(x.shape[:-2]))
        if not self.starting:
            out = out.transpose(-2, -3)
        return out

    T.TriangleAttention.forward = forward_xl
    log("tri_ln installed (TriangleAttention.forward: LN copy freed before chunked attention; per-chunk LN recompute)")
    return True


# ----------------------------------------------------------------------------------------------------------------- entry
_INSTALLED = set()


def install():
    req = requested()
    if not req:
        return
    unknown = req - KNOWN
    if unknown:
        log(f"unknown levers ignored: {sorted(unknown)}")
    if "tri_ln" in req and "tri_ln" not in _INSTALLED and _install_tri_ln():
        _INSTALLED.add("tri_ln")
    log(f"version {__version__}; installed={sorted(_INSTALLED)}; alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}")
    import atexit
    atexit.register(lambda: print(f"[odde_xl] STATS@exit {json.dumps(_STATS)}", file=sys.stderr, flush=True))
