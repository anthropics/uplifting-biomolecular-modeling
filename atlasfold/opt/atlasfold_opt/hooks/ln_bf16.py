"""Lever ln_bf16 — ``atlasfold.model.network.primitives.normalization:LayerNorm.forward`` (normalization.py L31-43). Stock computes
``F.layer_norm(x.float(), (C,), w.float(), b.float(), eps).to(x.dtype)``: on the bf16 trunk that is a bf16->fp32 copy, an fp32 LayerNorm and an
fp32->bf16 copy per call (~11 calls per PairBlock). torch's CUDA LayerNorm on a bf16 tensor with bf16 affine parameters computes the row
statistics in fp32 in-kernel, applies the affine in fp32 and rounds once to bf16 — the same arithmetic in one kernel, bit-identical to the stock
three-kernel statement on torch 2.7.1+cu128 (``vectorized_layer_norm_kernel``); the identity is a property of that torch build, not of LayerNorm in general.
Served: input bf16 on CUDA AND the module's weight / bias are bf16 (``lm_stack`` / ``main_stack``, which ``load_model`` casts with .to(bfloat16))
AND autocast disabled at the call (under the runner's bf16 autocast ``layer_norm`` is on the fp32 cast list and would re-create the stock path).
Every other call takes the stock statement, counted: ``fp32_input`` (fp32 activations: confidence / distogram / diffusion fp32 islands),
``fp32_params`` (a bf16 input reaching an fp32-parameter module — torch 2.7 refuses the mixed form, so it is never served), ``cpu``.
Class: exact (bitwise)."""
from . import Installed, rebind

TARGET = "atlasfold.model.network.primitives.normalization"
EXPECTED = ("fp32_input", "fp32_params", "cpu")


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    import torch
    import torch.nn.functional as Fn
    from opt_core.counters import Ledger
    try:
        nm = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001
        return Installed("ln_bf16", False, reason=f"import:{TARGET}:{type(e).__name__}")
    cls = nm.LayerNorm
    stock = cls.forward
    ledger = Ledger("LOCAL.atlasfold.ln_bf16", impl=f"F.layer_norm(bf16)@{torch.__version__.split('+')[0]}", origin="kit", expected=EXPECTED)
    bf16 = torch.bfloat16

    def forward(self, x):
        if not x.is_cuda:
            ledger.fallback("cpu")
            return stock(self, x)
        if x.dtype != bf16:
            ledger.fallback("fp32_input")
            return stock(self, x)
        w, b = self.weight, self.bias
        if (w is not None and w.dtype != bf16) or (b is not None and b.dtype != bf16):
            ledger.fallback("fp32_params")
            return stock(self, x)
        ledger.serve(f"C{self.normalized_shape}")
        with torch.autocast("cuda", enabled=False):                  # keep layer_norm off autocast's fp32 list: bf16 in, fp32 statistics in-kernel, bf16 out
            return Fn.layer_norm(x, (self.normalized_shape,), w, b, self.eps)
    forward.__qualname__ = "LayerNorm.forward[atlasfold_opt:ln_bf16]"
    rebind(cls, "forward", forward, stock)
    return Installed("ln_bf16", True, lines=[lambda: ledger.line(tag)], gates=[ledger.gate], facts={"impl": getattr(ledger, "impl", None)})
