"""thin — thin launchers for the ESM C eager path (pinned stack: TE 2.15 + flash-attn 2.7.4.post1 + Triton 3.6): the SAME kernels
with the SAME arguments, launched through the C++ bindings (transformer_engine_torch.layernorm_fwd / generic_gemm / swiglu,
flash_attn_2_cuda.varlen_fwd) and the cached Triton CudaLauncher directly, with the stock Python wrappers skipped.

The seam (esmc_opt.kits.fused._patch._OPS): after kit.apply(model):
    from esmc_opt.kits.thin import thin_launch
    _patch._OPS.update(thin_launch.ops(model))          # the entries: ln_qkv, out_proj, ffn, attn, rotary
Standalone on the stock model: apply(model) -> dict ; remove(model). No knobs.
"""
from . import thin_launch
from .thin_launch import TESTED_ITEMS, THIN_LAUNCH_VERSION, counters, disengage, engage, ops

KIT = "thin"


def apply(model):
    rec = engage(model)
    print(f"[{KIT}] thin launchers engaged: " + ", ".join(f"{k}={'ERROR ' + v['error'] if 'error' in v else 'ok'}" for k, v in rec["items"].items())
          + " — same kernels, same arguments through the C++ bindings / cached launcher; bitwise vs the stock")
    return rec


def remove(model):
    return disengage(model)
