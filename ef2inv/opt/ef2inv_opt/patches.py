"""The runtime patches an arm may carry, by name. (1) Every arm: the stock exception, upstream bug 0002 (STOCK.md 'Stock exception').

The fork's ``Transition._addmm_residual`` calls ``torch.addmm`` with operands of different dtypes (a bf16 activation, the fp32 ``w3``
weight) under ``set_kernel_backend('fused')``. On the pinned stack that is fatal in every arm: torch.compile's Inductor pad_mm pass
times the traced ``aten.addmm`` with those dtypes at the first design step ("self and mat2 must have the same dtype"), and at
``COMPILE=False`` the kit documents the same failure at the first confidence step. The one-line fix (``FUNCTION_TEXT``, applied here at
run time as a replacement of the fork's method) runs in every arm, stock included: the operands are
cast to the hidden activation's dtype before ``addmm`` and the result back to the input's — under the design step's bf16 autocast the
computation autocast performs anyway; in the fp32 confidence head no cast happens. It is applied unconditionally after the models are
loaded (no switch: the unpatched file cannot run on the pinned stack); the
function's sha256 and its source are recorded in ``run.json`` ``patches`` and ``opt_manifest.json`` ``deviations``.

(2) On request only — upstream fix EF2INV-0003 (``design | warm --upstream-fix EF2INV-0003``; ``upstream_issues/EF2INV-0003_esmc_rope_autograd.py``,
loaded by ``upstream_fix.py`` in the arm process, calls ``pin_esmc_rope()``). With flash-attn installed the fork's ``modeling_esmc``
applies ESM-C's rotary embedding with flash-attn's Triton kernel (``_flash_attn_rotary_available``, l.56-62; ``RotaryEmbedding.forward``
l.424-426 calls ``flash_attn.ops.triton.rotary.apply_rotary``, a raw launcher outside autograd — the rotated q/k carry no grad_fn, so the
pseudo-perplexity term's gradient, ``autograd.grad(plm_loss, logits)`` at stock file l.1094, loses its q·k path); without flash-attn it uses
the module's torch arithmetic (l.428-429), which autograd differentiates. ``pin_esmc_rope()`` (``ESMC_ROPE_FIX_IMPL``) rebinds that one
flag to False after the cookbook import so the arm keeps the differentiable RoPE while the folds run flash-attn. Without the flag no arm calls it and
ESM-C runs as imported. When applied it is a named deviation of the run (``run.json`` ``patches`` + ``upstream_fix``, ``opt_manifest.json``
``deviations`` + ``upstream_fix``) and the ATTN line's ``esmc_rope=`` word reads the flag back (attention.py).
"""
import hashlib
from typing import Optional

FIX_SOURCE = "STOCK.md 'Stock exception': the fork's Transition._addmm_residual with its addmm operands cast to the activation dtype, applied at run time as a replacement of the method"
FUNCTION_TEXT = '''def _addmm_residual_fixed(self, x, hidden):
    ffn = self.ffn; x_shape = x.shape
    x2 = x.contiguous().view(-1, x_shape[-1]); h2 = hidden.view(-1, hidden.shape[-1]); w = ffn.w3.weight; dt = h2.dtype
    out = torch.addmm(x2 if x2.dtype == dt else x2.to(dt), h2, w.t() if w.dtype == dt else w.t().to(dt))
    return (out if out.dtype == x.dtype else out.to(x.dtype)).view(x_shape)
'''
NAME = "upstream_bug_0002_transition_addmm_dtype"


def text_sha256() -> str:
    return hashlib.sha256(FUNCTION_TEXT.encode()).hexdigest()


def apply() -> dict:
    """Apply the fix to ``transformers.models.esmfold2.modeling_esmfold2_common.Transition`` (every arm, unconditionally); returns the record."""
    rec = {"name": NAME, "applied": True, "source": FIX_SOURCE,
           "function_sha256": text_sha256(), "target": "transformers.models.esmfold2.modeling_esmfold2_common.Transition._addmm_residual"}
    import torch
    from transformers.models.esmfold2 import modeling_esmfold2_common as C
    ns = {"torch": torch}
    exec(FUNCTION_TEXT, ns)
    rec["replaced_sha256"] = hashlib.sha256(C.Transition._addmm_residual.__code__.co_code).hexdigest()
    C.Transition._addmm_residual = ns["_addmm_residual_fixed"]
    return rec


# ---- (2) upstream fix EF2INV-0003, on request: ESM-C's rotary embedding pinned to the differentiable torch path
ESMC_ROPE_NAME = "esmc_rope_pin"
ESMC_ROPE_FIX_IMPL = "torch"                                               # the implementation upstream fix EF2INV-0003 pins (the record's `impl`); `flash_triton` = the word for the module as imported with flash-attn present
ESMC_ROPE_MODULE = "transformers.models.esmc.modeling_esmc"
ESMC_ROPE_FLAG = "_flash_attn_rotary_available"
ESMC_ROPE_SOURCE = f"{ESMC_ROPE_MODULE}.{ESMC_ROPE_FLAG} (l.56-62; read by RotaryEmbedding.forward l.424-429)"


def pin_esmc_rope(module=None) -> dict:
    """Pin ESM-C's RoPE to the fork's differentiable torch path on the imported ``modeling_esmc`` (``module``: injection for the CPU tests):
    the flag is rebound False (applied=True when it was True, i.e. flash-attn's Triton kernel was bound). Returns the record (``run.json`` ``patches``)."""
    impl = ESMC_ROPE_FIX_IMPL
    if module is None:
        import importlib
        module = importlib.import_module(ESMC_ROPE_MODULE)
    as_imported = bool(getattr(module, ESMC_ROPE_FLAG, False))
    rec = {"name": ESMC_ROPE_NAME, "impl": impl, "applied": False, "as_imported": as_imported, "source": ESMC_ROPE_SOURCE,
           "target": f"{ESMC_ROPE_MODULE}.{ESMC_ROPE_FLAG}"}
    if as_imported:
        setattr(module, ESMC_ROPE_FLAG, False); rec["applied"] = True
    rec["effective"] = "torch"
    return rec
