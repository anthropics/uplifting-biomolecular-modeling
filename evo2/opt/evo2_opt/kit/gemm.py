"""The GEMM-side levers on Transformer Engine's Linear: W1 (the FP8 weight workspace filled once and reused), E61 (general_gemm into a kept output buffer with cuBLASLt's own algorithm), the fp8 accumulation read-back, the census constants the route sets from the model (N_HYENA / N_ATTN / HIDDEN / WQKV_STRIDES)."""
from __future__ import annotations

import collections
import hashlib
import os

KIT = "v40_gemm_1"
ENGINE = "evo2_40b"
LEVERS = ("L2", "W1", "E50", "E56", "E61")
RUNGS = {"r1": ("L2",), "r2": ("L2", "W1"), "r3": ("L2", "W1", "E50"), "r4": ("L2", "W1", "E50", "E56"), "r5": ("L2", "W1", "E50", "E56", "E61")}
N_HYENA = 42          # evo2-40b-1m.yml L7-9: 14 hcl + 14 hcm + 14 hcs
N_ATTN = 8            # evo2-40b-1m.yml L10
HIDDEN = 8192         # evo2-40b-1m.yml L4
WQKV_STRIDES = (1, 3 * HIDDEN)   # the stock loader's column-split layout (vortex/model/model.py L1021-1031)
_HERE = os.path.dirname(os.path.abspath(__file__))
CTR = collections.Counter()


from evo2_opt.kit.errors import KitRefused   # noqa: E402


def expected_counts_static(levers) -> dict:
    """Per single forward at this module's census (N_HYENA Hyena blocks: the 40B's unless routes.configure set it); W1 counts are for a forward
    AFTER the cache-fill forward."""
    lv = set(levers)
    unknown = lv - set(LEVERS)
    if unknown:
        raise KitRefused(f"unknown levers {sorted(unknown)}; known {LEVERS}")
    c = {}
    if "W1" in lv:
        c["w1_weight_cache_hit"] = N_HYENA
    if "E56" in lv:
        c["e50_bmm_transposed"] = N_HYENA
        c["e56_bias_resid_fused"] = N_HYENA
    elif "E50" in lv:
        c["e50_bmm_transposed"] = N_HYENA
    if "E61" in lv:
        c["e61_out_buffer"] = N_HYENA
    return c


def te_forward_with_weight_cache(base_forward, counter: collections.Counter = CTR):
    """Wrapper for ``transformer_engine.pytorch.Linear.forward``: on a module flagged ``_v40_w1`` and called without an
    explicit ``is_first_microbatch`` (vortex's TELinear.forward, layers.py L82), pass True until the module holds its
    ``"weight"`` workspace (base.py L1107) and False afterwards; every other call is forwarded untouched."""
    def forward(self, inp, is_first_microbatch=None, *args, **kwargs):
        if is_first_microbatch is None and getattr(self, "_v40_w1", False):
            first = "weight" not in self._fp8_workspaces
            counter["w1_weight_cache_fill" if first else "w1_weight_cache_hit"] += 1
            is_first_microbatch = bool(first)
        return base_forward(self, inp, is_first_microbatch, *args, **kwargs)
    forward.__wrapped_stock__ = base_forward
    return forward


def _operand_shape(t):
    """A GEMM operand is a torch.Tensor, a Float8Tensor (torch.Tensor subclass) or — the cached weight workspace —
    a Float8TensorBase with no ``shape``: its row-wise uint8 ``_data`` carries the (rows, cols) shape."""
    if hasattr(t, "shape"):
        return tuple(t.shape)
    return tuple(t._data.shape)


def _operand_device(t):
    return t.device if hasattr(t, "device") else t._data.device


def general_gemm_with_out_buffer(base_general_gemm, counter: collections.Counter = CTR, buffers: dict | None = None):
    """Wrapper for ``transformer_engine.pytorch.module.linear.general_gemm`` (the name ``_Linear.forward`` calls,
    linear.py L53, L276-287): a plain bf16-output forward GEMM (``out`` None, no output quantizer, ``accumulate`` False,
    no comm overlap) gets a cached contiguous output buffer of the GEMM's shape — layout "TN": D = B @ A^T, shape
    (B.shape[0], A.shape[0]) (gemm.py L39, L51-53). Every other call is forwarded untouched."""
    bufs = {} if buffers is None else buffers
    workspaces = {}
    def general_gemm(A, B, workspace, out_dtype=None, quantization_params=None, gelu=False, gelu_in=None, accumulate=False,
                     layout="TN", out=None, bias=None, use_split_accumulator=False, grad=False, ub=None, ub_type=None, extra_output=None, bulk_overlap=False):
        if out is None and quantization_params is None and not accumulate and ub is None and not gelu and layout == "TN" and out_dtype is not None and getattr(general_gemm, "armed", False):
            import torch
            sa, sb = _operand_shape(A), _operand_shape(B)
            dev = _operand_device(B)
            key = (int(sb[0]), int(sa[0]), out_dtype, str(dev))
            out = bufs.get(key)
            if out is None:
                out = bufs[key] = torch.empty(key[0], key[1], dtype=out_dtype, device=dev)
            counter["e61_out_buffer"] += 1
            if workspace is not None and workspace.device != dev:                       # TE's module-level cuBLAS workspace lives on the device of its first
                ws = workspaces.get(str(dev))                                           # use; a GEMM on another device gets one of the same size on its own
                if ws is None:                                                          # device (the algorithm choice reads the size, not the address), so two
                    ws = workspaces[str(dev)] = torch.empty_like(workspace, device=dev)  # devices' GEMMs in flight at once never share scratch memory
                workspace = ws
                counter["e61_workspace_own_device"] += 1
        return base_general_gemm(A, B, workspace, out_dtype=out_dtype, quantization_params=quantization_params, gelu=gelu, gelu_in=gelu_in, accumulate=accumulate,
                                 layout=layout, out=out, bias=bias, use_split_accumulator=use_split_accumulator, grad=grad, ub=ub, ub_type=ub_type, extra_output=extra_output, bulk_overlap=bulk_overlap)
    general_gemm.__wrapped_stock__ = base_general_gemm
    general_gemm.buffers = bufs
    general_gemm.workspaces = workspaces
    general_gemm.armed = True
    return general_gemm


def accumulation_readback(model=None) -> dict:
    """The fp8 forward GEMM's accumulation mode READ BACK from the package: linear.py L270-274 takes
    ``recipe.fp8_gemm_fprop.use_split_accumulator`` when the recipe carries that attribute and ``_2X_ACC_FPROP``
    (base.py L49, imported into linear.py L22) otherwise; DelayedScaling (recipe/__init__.py L91-190) carries none."""
    import transformer_engine.pytorch.module.base as te_base
    import transformer_engine.pytorch.module.linear as te_linear
    out = {"base._2X_ACC_FPROP": getattr(te_base, "_2X_ACC_FPROP", None), "linear._2X_ACC_FPROP": getattr(te_linear, "_2X_ACC_FPROP", None)}
    recipe = None
    if model is not None:
        hb = [b for b in model.blocks if hasattr(b, "projections")]
        recipe = getattr(hb[0].projections, "fp8_recipe", None) if hb else None
        out["recipe_source"] = "model.blocks[].projections.fp8_recipe"
    if recipe is None:
        from transformer_engine.common.recipe import DelayedScaling
        recipe = DelayedScaling(); out["recipe_source"] = "DelayedScaling() default"
    out["recipe_class"] = type(recipe).__name__
    has = hasattr(recipe, "fp8_gemm_fprop"); out["recipe_has_fp8_gemm_fprop"] = bool(has)
    out["effective_use_split_accumulator"] = bool(recipe.fp8_gemm_fprop.use_split_accumulator) if has else out["linear._2X_ACC_FPROP"]
    return out


