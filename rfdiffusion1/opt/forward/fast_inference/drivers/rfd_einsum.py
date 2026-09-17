"""rfd_einsum.py — lever E (Tier 1, exact): route two opt_einsum contractions to torch.einsum with the SAME equation.

RFdiffusion binds `einsum = opt_einsum.contract` in Attention_module.py.  For two contraction signatures opt_einsum's pairwise path issues
extra permute/copy kernels around the same cuBLAS batched GEMM that torch.einsum reaches directly, and torch.einsum is the faster route on
the pinned stack (torch 2.4.0, opt_einsum 3.3.0) while producing byte-equal results:
    'bijh,bkjhd->bikhd'  BiasedAxialAttention (attn @ value; 76 calls/step)
    'bsqhd,bskhd->bqkh'  MSARowAttentionWithBias (q @ k; 32 calls/step)
All other signatures stay on opt_einsum (for 'bqkh,bskhd->bsqhd' and 'bihqk,bkihd->bqihd' torch.einsum is slower).

Exactness: the two backends can pick different kernels on another torch / opt_einsum version, so the driver re-checks op-level byte
equality of both routed signatures on the live stack at start-up (on the model's own shapes) and refuses the lever by name if it does
not hold; with the check passed the routed contractions are the same GEMMs on the same values.
apply() patches the module-level `einsum` symbol used by Attention_module with a dispatcher.
"""
import torch

ROUTE_TO_TORCH = {"bijh,bkjhd->bikhd", "bsqhd,bskhd->bqkh"}
STATS = dict(torch_routed=0, opt_einsum=0)
_oe = None


def _dispatch(eq, *ops, **kw):
    if eq in ROUTE_TO_TORCH and not kw:
        STATS["torch_routed"] += 1
        return torch.einsum(eq, *ops)
    STATS["opt_einsum"] += 1
    return _oe(eq, *ops, **kw)


def live_check(device="cuda"):
    """op-level byte-equal check on this stack for the routed signatures at representative shapes; returns (ok, rows)."""
    import rfdiffusion.Attention_module as AM
    oe = AM.einsum if _oe is None else _oe
    rows, ok = [], True
    g = torch.Generator(device=device).manual_seed(0)
    for L in (64, 195, 291):
        a = torch.softmax(torch.randn(1, L, L, 4, device=device, generator=g), dim=-2); v = torch.randn(1, L, L, 4, 32, device=device, generator=g)
        r1 = oe("bijh,bkjhd->bikhd", a, v); r2 = torch.einsum("bijh,bkjhd->bikhd", a, v)
        q = torch.randn(1, 1, L, 8, 32, device=device, generator=g); k = torch.randn(1, 1, L, 8, 32, device=device, generator=g)
        s1 = oe("bsqhd,bskhd->bqkh", q, k); s2 = torch.einsum("bsqhd,bskhd->bqkh", q, k)
        b1, b2 = bool(torch.equal(r1, r2)), bool(torch.equal(s1, s2))
        rows.append(dict(L=L, av_bitwise=b1, qk_bitwise=b2)); ok = ok and b1 and b2
    return ok, rows


def apply(require_live_check=True):
    global _oe
    import rfdiffusion.Attention_module as AM
    from opt_einsum import contract
    if AM.einsum is _dispatch:
        return "already applied"
    assert AM.einsum is contract, "Attention_module.einsum is not opt_einsum.contract (another patch active?)"
    _oe = AM.einsum
    if require_live_check:
        ok, rows = live_check()
        if not ok:
            _oe = None
            raise RuntimeError(f"lever E refused: torch.einsum not bitwise with opt_einsum on this stack: {rows}")
    AM.einsum = _dispatch
    return f"lever E active: {sorted(ROUTE_TO_TORCH)} -> torch.einsum (live op-level check bitwise)"


def stats():
    return dict(STATS)
