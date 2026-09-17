"""Row-blocked LayerNorm launch (lever ``ln_rows``): xfold's fastnn Triton LayerNorm kernel, ROWS rows per program instead of one.

Stock (``xfold/fastnn/layer_norm.py``, the path every mode runs wherever a kernel lever has not absorbed the LayerNorm — under ``off`` /
``exact`` all 773 of the model's LayerNorms): ``LayerNorm.forward`` → ``LayerNormTritonFunc.apply`` → the Triton kernel
``_layer_norm_fwd_fused`` launched with ONE PROGRAM PER ROW (grid ``(M,)``): fp32 two-pass mean / variance, ``1 / tl.sqrt``, the affine
map, the result stored in the input's dtype. On the pair track a row is 128 bf16 values (256 bytes) and a call at 800 tokens is 640 000
one-warp programs: the launch, not the arithmetic, is the cost (the call runs at a fraction of the card's bandwidth; thousands of calls
per item in the ``exact`` trunk, and more on the atom-pair rows inside the sampler's step graphs).

Here (``_layer_norm_fwd_rows`` below): the SAME kernel body — statement for statement, pinned to the kit's source by tests/test_ln_rows.py —
inside a loop over ``ROWS`` consecutive rows of one program (grid ``(cdiv(M, ROWS),)``), with the stock ``BLOCK_SIZE`` / ``num_warps``
choice. Each row is computed by the identical 1-D code (the same layouts, the same reduction tree, the same conversions), so every output
element is the stock kernel's bit for bit — bitwise by construction, and measured so at every (width, dtype, affine, contiguity) the model
calls it with (16 | 64 | 128 | 256 | 267 | 384 | 768 | 833 columns; bf16 and fp32 rows; odd row counts). Rows of 1 KB and wider (fp32 at
256 columns, the 267-wide diffusion pair conditioning, …) gain nothing from blocking and keep the stock launch, counted (``wide``); so do
CPU tensors and the ``torch`` implementation setting (``generic``: the stock statement as it is). The census (``COUNTS``) rides
forward.json (``ln_rows``) and the LEVER line.
"""
from __future__ import annotations

COUNTS = {"calls": 0, "served": 0, "wide": 0, "generic": 0}
ROWS = 8                          # rows per program: 4 / 8 / 16 / 32 measure within 3 % of each other at the pair cells (H100); 8 keeps small calls' grids wide
MAX_ROW_BYTES = 512               # rows this wide or narrower (BLOCK_SIZE * element size) are served; wider rows keep the stock one-row launch (measured: no gain at 1 KB rows)
_STATE = {"kernel": None}


def take() -> dict:
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    return out


def _kernel():
    """The row-blocked kernel, built on first use (Triton imported in the model process only)."""
    if _STATE["kernel"] is not None:
        return _STATE["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def _layer_norm_fwd_rows(X, Y, W, B, M, N, eps, ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr, USE_WEIGHTS: tl.constexpr = True, USE_BIAS: tl.constexpr = True):
        pid = tl.program_id(0).to(tl.int64)
        for r in range(ROWS):
            row = pid * ROWS + r                  # int64, as the stock kernel's `row` (rows * N passes 2**31 - 1 on the diffusion pair conditioning past 2836 tokens)
            if row < M:
                Xr = X + row * N                  # stock: X += row * N; Y += row * N — the row's base pointers
                Yr = Y + row * N
                # ---- the stock kernel body (xfold/fastnn/layer_norm.py _layer_norm_fwd_fused), X / Y spelt Xr / Yr ----
                # Compute mean
                mean = 0
                _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    cols = off + tl.arange(0, BLOCK_SIZE)
                    a = tl.load(Xr + cols, mask=cols < N, other=0.).to(tl.float32)
                    _mean += a
                mean = tl.sum(_mean, axis=0) / N
                # Compute variance
                _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    cols = off + tl.arange(0, BLOCK_SIZE)
                    x = tl.load(Xr + cols, mask=cols < N, other=0.).to(tl.float32)
                    x = tl.where(cols < N, x - mean, 0.)
                    _var += x * x
                var = tl.sum(_var, axis=0) / N
                rstd = 1 / tl.sqrt(var + eps)
                # Normalize and apply linear transformation
                for off in range(0, N, BLOCK_SIZE):
                    cols = off + tl.arange(0, BLOCK_SIZE)
                    mask = cols < N
                    if USE_WEIGHTS is True:
                        w = tl.load(W + cols, mask=mask)
                        if USE_BIAS is True:
                            b = tl.load(B + cols, mask=mask)
                    x = tl.load(Xr + cols, mask=mask, other=0.).to(tl.float32)
                    x_hat = (x - mean) * rstd
                    if USE_WEIGHTS is True:
                        y = x_hat * w
                        if USE_BIAS is True:
                            y += b
                    else:
                        y = x_hat
                    # Write output
                    tl.store(Yr + cols, y, mask=mask)

    _STATE["kernel"] = _layer_norm_fwd_rows
    return _layer_norm_fwd_rows


def layer_norm_rows(x, normalized_shape, weight, bias, eps):
    """``LayerNormTritonFunc.forward``'s statements with the row-blocked launch: y = LayerNorm(x) over the last dim, y in x's dtype.
    Returns None (nothing launched) for rows wider than ``MAX_ROW_BYTES``: the caller runs the stock launch."""
    import triton
    x = x.contiguous()
    N = x.shape[-1]
    M = x.numel() // N
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    if BLOCK_SIZE * x.element_size() > MAX_ROW_BYTES or M == 0:
        return None
    y = torch_empty_like(x)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    rows = min(ROWS, M)
    _kernel()[(triton.cdiv(M, rows),)](x, y, weight, bias, M, N, eps, ROWS=rows, BLOCK_SIZE=BLOCK_SIZE, USE_WEIGHTS=weight is not None, USE_BIAS=bias is not None,
                                        num_warps=num_warps, num_ctas=1)
    return y


def torch_empty_like(x):
    import torch
    return torch.empty_like(x)


_BOUND = {"lnm": None, "torch": None}     # xfold.fastnn.layer_norm and torch, bound once by install (no import statement on the per-call path: the diffusion module's compiled blocks trace this forward)


def _modules():
    if _BOUND["lnm"] is None:
        import importlib
        import torch
        _BOUND["lnm"] = importlib.import_module("xfold.fastnn.layer_norm"); _BOUND["torch"] = torch
    return _BOUND["lnm"], _BOUND["torch"]


def forward(self, input):
    """``xfold.fastnn.layer_norm.LayerNorm.forward`` restated: the ``triton`` implementation's CUDA rows go to the row-blocked launch; the
    ``torch`` implementation (what ``fast`` / ``big`` run), CPU tensors and wide rows run the stock statement, counted. Inside a
    torch.compile trace (the diffusion module's compiled blocks under ``compile``) the census statements drop out (``is_compiling``), so the
    traced code is the stock statement alone."""
    lnm, torch = _modules()
    F = torch.nn.functional
    fastnn_config = lnm.fastnn_config
    counting = not torch.compiler.is_compiling()
    if counting:
        COUNTS["calls"] += 1
    if fastnn_config.layer_norm_implementation == "torch":                     # stock statement (the eager setting: fast / big outside the kernel levers' reach, --nofastnn)
        if counting:
            COUNTS["generic"] += 1
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
    elif fastnn_config.layer_norm_implementation == "triton":
        y = layer_norm_rows(input, self.normalized_shape, self.weight, self.bias, self.eps) if input.is_cuda else None
        if y is not None:
            if counting:
                COUNTS["served"] += 1
            return y
        if counting:
            COUNTS["wide" if input.is_cuda else "generic"] += 1
        LayerNormTritonFunc = lnm.LayerNormTritonFunc
        fast_out = LayerNormTritonFunc.apply(                                  # stock statement: the one-row-per-program launch
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return fast_out
    else:
        raise ValueError(
            f"fastnn_config must be 'torch' or 'triton', got {fastnn_config}"
        )


def _is_stock_forward(f, lnm) -> bool:
    return getattr(f, "__module__", None) == lnm.__name__ and getattr(f, "__qualname__", "") == "LayerNorm.forward"


def install(model) -> dict:
    """Bind ``forward`` on the fastnn LayerNorm CLASS when its forward is still the kit's own (no other lever rebound it); else step aside,
    named. Idempotent. {'installed', 'already', 'modules', 'reason'} — ``modules``: the model's fastnn LayerNorm count (what the class
    binding reaches; kernel levers that absorb a LayerNorm into a fused producer never call it, so ``served`` counts what actually ran)."""
    lnm, _ = _modules()
    LN = lnm.LayerNorm
    n = sum(1 for m in model.modules() if isinstance(m, LN))
    if getattr(LN, "_ln_rows", None):
        return {"installed": True, "already": True, "modules": n, "reason": getattr(LN, "_ln_rows_reason", None)}
    for name in ("normalized_shape", "weight", "bias", "eps"):
        probe = next((m for m in model.modules() if isinstance(m, LN)), None)
        if probe is not None and not hasattr(probe, name):
            raise RuntimeError(f"ln_rows: fastnn LayerNorm has no {name!r} (the kit's module changed shape)")
    reason = None
    if _is_stock_forward(LN.forward, lnm):
        LN._stock_forward = LN.forward
        LN.forward = forward
        installed = True
    else:
        installed = False; reason = "class_forward_rebound_by_another_lever"
    LN._ln_rows = installed or "aside"; LN._ln_rows_reason = reason
    return {"installed": installed, "already": False, "modules": n, "reason": reason}
