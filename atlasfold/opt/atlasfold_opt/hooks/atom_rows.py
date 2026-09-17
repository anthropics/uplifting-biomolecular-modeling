"""Lever atom_rows (class fast: tolerance) — the atom blocks' CONDITIONED TRANSITION on opt_core's fused row kernels.

Stock: every atom block of the denoiser's AtomEncoder / AtomDecoder (diffusion_transformer.py AtomTransformerBlock L354-360: 3 blocks x 2
stacks per denoiser call, 200 calls per roll-out) ends in ConditionedTransitionBlock.forward (L54-59) on the [B, N, W, 56, 96] query rows:
`a = self.adaln(a, cond)` (normalization.py L71-76: an fp32 LayerNorm kernel, then `sigmoid(linear_g(LN(cond))) * a + linear_bias(LN(cond))` =
a multiply and an add pass — the cond factors are roll-out constants sampler_hoist memoises), `b = self.swiglu(a)` (one GEMM 96 -> 384, then
`silu(x1) * x2`: two elementwise passes over the [.., 384] rows), `a = sigmoid(self.linear_g(cond)) * self.linear_out(b)` (one GEMM, a sigmoid
and a multiply pass).  Eight elementwise / normalisation kernels per block around three GEMMs, each a full read + write of the fp32 atom rows
(24 MB per [5, 224*56, 96] tensor at the 896 bucket) — among the largest non-GEMM, non-attention kernels of the atom path.

This lever: the same three statements as THREE Triton row kernels of opt_core (opt_core.kernels.dtk_kernels; fp32 statistics and arithmetic,
one rounding to the output dtype):
    y  = ln_modulate(a_rows, scale=sig, shift=bias, sigmoid_scale=False, mod_period=P)   LayerNorm (no affine, the AdaLN's eps) + the AdaLN
                                                                                        combine in ONE pass; sig / bias = the cond-side factors
                                                                                        (sigmoid(linear_g(LN(cond))), linear_bias(LN(cond))) as
                                                                                        [W*56, c] rows serving row r of the [N*W*56, c] sample
                                                                                        rows by r % P (the conditioning is one row per atom for
                                                                                        all N samples — never expanded over N)
    ab = self.swiglu.linear(y);  h = swiglu(ab)                                          the stock GEMM, then silu(x1) * x2 in ONE pass
    o  = self.linear_out(h);     a = gate_residual(o, gate=linear_g(cond) rows, sigmoid) the stock GEMM, then sigmoid(gate) * o in ONE pass
The cond-side factors join sampler_hoist's roll-out memo when it holds `cond` as an invariant (counted leaf_memo_hit / leaf_memo_miss; freed when
DiffusionHead.sample returns), else run per call (leaf_per_call); `self.linear_g(cond)` is the stock module call (sampler_hoist's instance memo
answers it inside a roll-out).  dtype flow = the stock flow: outside autocast fp32 throughout; under atom_bf16's window the modulated rows are
emitted in the autocast dtype (what the following Linear would cast them to), the GEMMs run as torch runs them, the row kernels compute in fp32.
Tolerance class: the Triton LayerNorm / sigmoid / silu differ from torch's kernels in summation order and by a few ulp — never bitwise;
deterministic (no atomics).  Composes with atom_kdedup (a different
site), atom_tf32 / atom_bf16 (this body runs inside their windows), sampler_hoist (its memos are read through the modules it patched, its
AdaLN instance forward on `self.adaln` is simply not called), denoiser_graph (Triton kernels compile at the eager warm-up call and are captured).
Guards, by name: AFO_ATOM_ROWS=0 -> `disabled`; CPU tensors -> `cpu`; operands that are not the rank-5 atom rows (the token transformer's own
ConditionedTransitionBlock calls are rank 4) -> `rank` (expected: those calls keep the statement below by design); a conditioning tensor whose
window geometry is not the rows' (`cond_form`) or a core without the row kernels / triton (`no_kernels`) -> the statement below (unexpected: gate refused)."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from . import Installed, rebind, size_gated

NAME = "LOCAL.atlasfold.atom_rows"
ENV = "AFO_ATOM_ROWS"
TARGET = "atlasfold.model.network.diffusion_transformer"                       # ConditionedTransitionBlock.forward
T_NORM = "atlasfold.model.network.primitives.normalization"
IMPL = "dtk_kernels(ln_modulate+swiglu+gate_residual)"
EXPECTED = ("disabled", "cpu", "rank")
SOURCE_SHA256 = {
    "ConditionedTransitionBlock.forward": ("5a3598684804a2dc0b11cc8fbb3b16ce8bae62574414f3e186faccc4e291fd15",),
    "AdaLN.forward": ("2365a0ab8c97565094c686043c4e7d46c14b3b9effc10c554c695efbd4b8f680",),
}
_STATE = {"override": None}
_K = {"mod": None, "err": None}


def kernels():
    """opt_core.kernels.dtk_kernels, imported at the first CUDA call (Triton is only needed on a CUDA host: a CPU-only process never reaches this)."""
    if _K["mod"] is None and _K["err"] is None:
        try:
            from opt_core.kernels import dtk_kernels
            _K["mod"] = dtk_kernels
        except Exception as e:  # noqa: BLE001
            _K["err"] = f"{type(e).__name__}"
    return _K["mod"]


def enabled() -> bool:
    if _STATE["override"] is not None:
        return bool(_STATE["override"])
    return os.environ.get(ENV, "1") != "0"


def bench_arm(label: str) -> None:
    """In-process override of AFO_ATOM_ROWS (not a kit switch): label 'A' = the statement below (lever inert), any other label = the row kernels."""
    _STATE["override"] = (label != "A")


bench_arm.state = lambda: {"atom_rows": enabled()}   # type: ignore[attr-defined]


def _rollout():
    try:
        from . import sampler_hoist as SH
        return SH._cur()
    except Exception:  # noqa: BLE001
        return None


def _ada_leaves(ada, cond: torch.Tensor, ledger) -> Tuple[torch.Tensor, torch.Tensor]:
    """(sigmoid(linear_g(layernorm_cond(cond))), linear_bias(layernorm_cond(cond))) — normalization.py L74 + the cond-only factors of L76;
    memoised in sampler_hoist's roll-out memo when it holds `cond`, else per call."""
    ro = _rollout()
    if ro is not None and "atom_cond" in getattr(ro, "hoists", ()) and id(cond) in getattr(ro, "inv_ids", ()):
        key = ("atom_rows.ada", id(ada), id(cond))
        sb = ro.memo.get(key)
        if sb is None:
            c = ada.layernorm_cond(cond)
            sb = (ada.sigmoid(ada.linear_g(c)).contiguous(), ada.linear_bias(c).contiguous())
            ro.memo[key] = sb
            ro.hold(*sb)
            ledger.count("leaf_memo_miss")
        else:
            ledger.count("leaf_memo_hit")
        return sb
    c = ada.layernorm_cond(cond)
    ledger.count("leaf_per_call")
    return ada.sigmoid(ada.linear_g(c)).contiguous(), ada.linear_bias(c).contiguous()


def source_check() -> Tuple[Optional[str], dict]:
    import importlib
    from opt_core.diffusion_loop.source_guard import source_sha256
    DT = importlib.import_module(TARGET); norm = importlib.import_module(T_NORM)
    seen = {}
    cur = DT.ConditionedTransitionBlock.forward
    if hasattr(cur, "__wrapped_stock__"):
        return f"wrapped:{getattr(cur, '__qualname__', '?')}", seen
    ada_f = norm.AdaLN.forward
    while hasattr(ada_f, "__wrapped_stock__"):
        ada_f = ada_f.__wrapped_stock__
    for fn, f in (("ConditionedTransitionBlock.forward", cur), ("AdaLN.forward", ada_f)):
        try:
            d = source_sha256(f)
        except Exception as e:  # noqa: BLE001
            d = f"unreadable:{type(e).__name__}"
        seen[fn] = d
        if d not in SOURCE_SHA256[fn]:
            return f"source:{fn}", seen
    return None, seen


def make_forward(stock, ledger):
    def forward(self, a, cond):
        if not enabled():
            ledger.fallback("disabled")
            return stock(self, a, cond)
        if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
            ledger.fallback("cpu")
            return stock(self, a, cond)
        if a.dim() != 5 or not isinstance(cond, torch.Tensor) or cond.dim() != 5:
            ledger.fallback("rank")
            return stock(self, a, cond)
        K = kernels()
        if K is None:                                                          # a CUDA host whose opt_core lacks the row kernels / triton: the statement below, gate refused
            ledger.fallback("no_kernels")
            return stock(self, a, cond)
        B, N, W, L, c = (int(x) for x in a.shape)
        Bc, Nc = int(cond.shape[0]), int(cond.shape[1])
        if tuple(int(x) for x in cond.shape[2:4]) != (W, L) or Bc not in (1, B) or Nc not in (1, N):
            ledger.fallback("cond_form")
            return stock(self, a, cond)
        ada = self.adaln
        sig, bias = _ada_leaves(ada, cond, ledger)                              # [Bc, Nc, W, L, c] each (post-sigmoid scale, shift)
        g = self.linear_g(cond)                                                # [Bc, Nc, W, L, c] pre-sigmoid gate (sampler_hoist's memo inside a roll-out)
        if not g.is_contiguous():
            g = g.contiguous()
        ydt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else a.dtype   # what the next Linear would cast the modulated rows to
        a_ = a if a.is_contiguous() else a.contiguous()
        R1 = N * W * L                                                         # sample rows per batch element
        P = W * L if Nc == 1 else R1                                           # modulation period: one conditioning row per atom serves all N samples
        eps = float(getattr(ada.layernorm, "eps", 1e-5))
        outs = []
        for b in range(B):                                                     # B = 1 above the 512 bucket, 2 at it: the conditioning block of each element is its own period
            xb = a_[b].reshape(R1, c)
            cb = b if Bc == B else 0
            scb, shb, gb = sig[cb].reshape(-1, c), bias[cb].reshape(-1, c), g[cb].reshape(-1, c)
            # a = self.adaln(a, cond)                                           normalization.py L71-76 in one pass
            y = K.ln_modulate(xb, scale=scb, shift=shb, eps=eps, out_dtype=ydt, sigmoid_scale=False, mod_period=P)
            # b = self.swiglu(a)                                                activation.py L13-15: the GEMM, then silu(x1) * x2 in one pass
            ab = self.swiglu.linear(y)
            h = K.swiglu(ab.reshape(R1, -1))
            # a = torch.sigmoid(self.linear_g(cond)) * self.linear_out(b)      the GEMM, then the sigmoid gate in one pass
            o = self.linear_out(h)
            outs.append(K.gate_residual(o.reshape(R1, c), gate=gb, gate_period=P, sigmoid_gate=True).view(N, W, L, c))
        out = outs[0].unsqueeze(0) if B == 1 else torch.stack(outs, 0)
        ledger.serve(f"W{W}xN{N}x{L}")
        return out
    forward.__qualname__ = "ConditionedTransitionBlock.forward[atlasfold_opt:atom_rows]"
    return forward


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    from opt_core.counters import Ledger
    try:
        DT = importlib.import_module(TARGET)
        importlib.import_module(T_NORM)
    except Exception as e:  # noqa: BLE001
        return Installed("atom_rows", False, reason=f"import:{type(e).__name__}")
    why, seen = source_check()
    if why is not None:
        return Installed("atom_rows", False, reason=why, facts={"digests": seen})
    ledger = Ledger(NAME, impl=IMPL, origin="kit", expected=EXPECTED)
    for k in ("leaf_memo_hit", "leaf_memo_miss", "leaf_per_call"):
        ledger.set(k, 0)
    cls = DT.ConditionedTransitionBlock
    stock = cls.forward
    rebind(cls, "forward", make_forward(stock, ledger), stock)

    def restore() -> None:
        if getattr(cls.forward, "__wrapped_stock__", None) is stock:
            setattr(cls, "forward", stock)

    def line():
        ev = {}
        if not enabled():
            ev["switch"] = f"{ENV}=0"
        return ledger.line(tag, **ev)
    return Installed("atom_rows", True, lines=[line], gates=[size_gated(ledger)],
                     facts={"impl": IMPL, "env": os.environ.get(ENV, "1"), "digests": seen, "ledger": ledger, "restore": restore})
