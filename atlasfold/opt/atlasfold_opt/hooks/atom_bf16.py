"""Lever atom_bf16 (class fast: tolerance) — bf16 autocast SCOPED to the atom-attention encoder / decoder of the diffusion denoiser.

Stock: DiffusionModule.forward (diffusion_head.py L122-181) runs the AtomEncoder (L151) and the AtomDecoder (L178) inside
`torch.autocast(..., enabled=False)` islands: the two 3-block windowed atom transformers (AtomAttentionStack.forward, atom_attention.py
L44-100: c_atom 96 = 2 heads x 48, query rows [N, W, 56, 96], key / value rows [N, W, 168, 96], SwiGLU 96 -> 384 -> 96, AdaLN, the
pair-bias projection linear_pair_bais 14 -> blocks x heads) run in fp32 while the 12-block token transformer between them runs under the
sampler's bf16 autocast (diffusion_bf16).  The two fp32 islands are the larger part of a denoiser call: GEMM time plus fp32 activation
traffic (LayerNorm / elementwise / layout passes).

This lever: AtomAttentionStack.forward — called exactly twice per denoiser call (AtomEncoder.forward L169, AtomDecoder.forward L202), nowhere
else — runs under `torch.autocast(device_type, dtype=torch.bfloat16)`; its result is returned in the caller's dtype (the residual stream
`a_q + attention(...)` stays fp32 by type promotion, so the cast is a no-op guard).  Inside: the Linear GEMMs take bf16 operands with fp32
accumulation (q/k/v/gate/out, AdaLN linear_g / linear_bias, SwiGLU, linear_pair_bais), LayerNorm statistics stay fp32 (upstream's LayerNorm
computes in float and autocast keeps layer_norm fp32), softmax / SDPA follow their operands (atom_sdpa: the fused kernel's bf16 route, bias
buffer in bf16; the MATH statement: bf16 bmm).  KEEP_COORD (as atom_tf32): the coordinate-facing precision=32 linears sit OUTSIDE the call by
construction — AtomEncoder.linear_in (reads r_noisy) before the stack, AtomDecoder.linear_out (writes r_update) after it, random_augmentation and
the roll-out arithmetic outside DiffusionModule.forward; the one coordinate-reading producer that CAN run inside (relpos_lazy replays
AtomRelativePositionEncoding.forward at its consumer, inside sampler_hoist's atom_pair hoist or per call) is wrapped to run with autocast OFF
(counted `guard_relpos`).  Autocast is host state consulted when an op is issued: under denoiser_graph it is on during the warm-up call and the
CAPTURE, so the bf16 kernels are what the graph records and replays (census = warm-up + capture calls, like every lever inside the graph).
Composes with atom_tf32 (bf16 GEMMs ignore allow_tf32; the flag still governs any fp32 GEMM left inside), atom_sdpa, atom_kdedup, sampler_hoist
(installed after them: this window is the outermost wrapper of AtomAttentionStack.forward, so the hoisted prologue and the row-form key side run
inside it exactly as the per-step body does).  Tolerance class, deterministic, never bitwise to --mode off.
AFO_ATOM_BF16=0: the lever installs and every call runs at the caller's dtype policy, counted `disabled`; MODEL_OPT_LEVERS_OFF=atom_bf16 removes
it from the row.  CPU tensors: counted `cpu`, stock."""
from __future__ import annotations

import contextlib
import os

import torch

from . import Installed, rebind, size_gated

NAME = "LOCAL.atlasfold.atom_bf16"
ENV = "AFO_ATOM_BF16"
TARGET = "atlasfold.model.network.atom_attention"
PRODUCER = "atlasfold.model.network.rel_pos_encoding"
EXPECTED = ("disabled", "cpu")
DTYPE = torch.bfloat16
_STATE = {"depth": 0, "override": None}          # override: None = read AFO_ATOM_BF16; True/False = forced in-process by bench_arm()


def enabled() -> bool:
    if _STATE["override"] is not None:
        return bool(_STATE["override"])
    return os.environ.get(ENV, "1") != "0"


def bench_arm(label: str) -> None:
    """In-process override of AFO_ATOM_BF16 (not a kit switch): label 'A' = the caller's dtype policy (lever inert), any other label = the bf16 window."""
    _STATE["override"] = (label != "A")


bench_arm.state = lambda: {"atom_bf16": enabled(), "depth": _STATE["depth"]}   # type: ignore[attr-defined]


@contextlib.contextmanager
def window(device_type: str = "cuda"):
    """bf16 autocast for the body (depth-counted so the producer guard knows it is inside)."""
    _STATE["depth"] += 1
    try:
        with torch.autocast(device_type=device_type, dtype=DTYPE):
            yield
    finally:
        _STATE["depth"] -= 1


def _device_of(args, kwargs, default: str = "cuda") -> str:
    """Device type of the first tensor among the call's arguments (a features dict is looked into), else `default`."""
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.Tensor):
            return a.device.type
        if isinstance(a, dict):
            for v in a.values():
                if isinstance(v, torch.Tensor):
                    return v.device.type
    return default


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    from opt_core.counters import Ledger
    try:
        AA = importlib.import_module(TARGET)
        RP = importlib.import_module(PRODUCER)
    except Exception as e:  # noqa: BLE001
        return Installed("atom_bf16", False, reason=f"import:{type(e).__name__}")
    ledger = Ledger(NAME, impl="autocast.bf16(AtomAttentionStack.forward)", origin="kit", expected=EXPECTED)
    ledger.set("guard_relpos", 0)
    cls = AA.AtomAttentionStack
    stock = cls.forward                                                          # AtomAttentionStack.forward at install time — in the fast row: atom_tf32's window over sampler_hoist's atom_pair wrapper over upstream's

    def forward(self, batch, q, *args, **kwargs):
        if not enabled():
            ledger.fallback("disabled")
            return stock(self, batch, q, *args, **kwargs)
        if q.device.type != "cuda":
            ledger.fallback("cpu")
            return stock(self, batch, q, *args, **kwargs)
        with window(q.device.type):
            out = stock(self, batch, q, *args, **kwargs)
        ledger.serve("capture" if torch.cuda.is_current_stream_capturing() else "eager")
        return out if out.dtype == q.dtype else out.to(q.dtype)
    forward.__qualname__ = "AtomAttentionStack.forward[atlasfold_opt:atom_bf16]"
    rebind(cls, "forward", forward, stock)

    rp_cls = RP.AtomRelativePositionEncoding
    rp_stock = rp_cls.forward

    def rp_forward(self, *args, **kwargs):                                      # KEEP_COORD guard: the reference-atom offset table is built from coordinates at the caller's precision, autocast off
        if _STATE["depth"] > 0:
            dev = _device_of(args, kwargs)
            if torch.is_autocast_enabled(dev):
                ledger.count("guard_relpos")
                with torch.autocast(device_type=dev, enabled=False):
                    return rp_stock(self, *args, **kwargs)
        return rp_stock(self, *args, **kwargs)
    rp_forward.__qualname__ = "AtomRelativePositionEncoding.forward[atlasfold_opt:atom_bf16.guard]"
    rebind(rp_cls, "forward", rp_forward, rp_stock)

    def restore() -> None:
        if getattr(cls.forward, "__wrapped_stock__", None) is stock:
            setattr(cls, "forward", stock)
        if getattr(rp_cls.forward, "__wrapped_stock__", None) is rp_stock:
            setattr(rp_cls, "forward", rp_stock)

    def line():
        ev = {}
        if not enabled():
            ev["switch"] = f"{ENV}=0"
        return ledger.line(tag, **ev)
    return Installed("atom_bf16", True, lines=[line], gates=[size_gated(ledger)],
                     facts={"impl": getattr(ledger, "impl", None), "env": os.environ.get(ENV, "1"), "ledger": ledger, "restore": restore})
