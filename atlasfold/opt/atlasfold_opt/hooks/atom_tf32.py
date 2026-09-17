"""Lever atom_tf32 (class fast: tolerance) — TF32 tensor-core GEMMs SCOPED to the atom-attention encoder / decoder of the diffusion denoiser.

Stock: DiffusionModule.forward (diffusion_head.py L122-181) runs the AtomEncoder (L151) and the AtomDecoder (L178) with autocast DISABLED while
the process sits at torch.backends.cuda.matmul.allow_tf32 = False (cli/multimer.py L402 sets float32_matmul_precision 'highest'; the det recipe
sets the same): every Linear of the two 3-block windowed atom transformers (c_atom 96 = 2 heads x 48; query rows [N, W, 56, 96], key / value
rows on the 3x-unfolded key windows [N, W, 168, 96]), the pair-bias projection linear_pair_bais (14 -> blocks x heads) and the two bmm of the
MATH-backend SDPA are IEEE-fp32 GEMMs on the CUDA cores.  The atom encoder + decoder are the larger part of a denoiser call, and these GEMMs
a large share of the two.

This lever: AtomAttentionStack.forward (atom_attention.py L44-100) — called exactly twice per denoiser call, by AtomEncoder.forward (L128) and
AtomDecoder.forward (L196), nowhere else in the model — runs inside a host-side precision WINDOW: allow_tf32 := True on entry, the entry value
restored on exit (try/finally; nested entries restore what they found).  KEEP_COORD: the coordinate-facing
linears sit OUTSIDE that call by construction — AtomEncoder.linear_in (precision=32, reads r_noisy; L127) runs before the stack,
AtomDecoder.linear_out (LayerNorm + precision=32 Linear, writes r_update; L197) after it, DiffusionHead.random_augmentation and the roll-out
arithmetic outside DiffusionModule.forward — so they need no instance guard; the one fp32 producer that CAN run inside the window (relpos_lazy
replays AtomRelativePositionEncoding.forward at its consumer) is wrapped to run at allow_tf32 = False (guard, counted `guard_relpos`).
The window is host state that cuBLAS reads when a GEMM is issued: under denoiser_graph it is open during the warm-up call and the CAPTURE, so the
TF32 kernels are what the graph records and every replay runs (the flag is not consulted at replay; the census then counts warm-up + capture
calls only, like every lever inside the graph).  TF32 = 10-bit-mantissa products with fp32 accumulation: tolerance class,
deterministic, never bitwise to --mode off; the bf16 DiT between the two islands is untouched (autocast bf16 GEMMs ignore the flag).
AFO_ATOM_TF32=0: the lever installs and every call runs at the entry precision, counted `disabled` (the kit's AFO_* convention);
MODEL_OPT_LEVERS_OFF=atom_tf32 removes it from the row before activation.  CPU tensors / pre-Ampere parts: counted `cpu` / `cc<8`, stock."""
from __future__ import annotations

import contextlib
import os

import torch

from . import Installed, rebind, size_gated

NAME = "LOCAL.atlasfold.atom_tf32"
ENV = "AFO_ATOM_TF32"
TARGET = "atlasfold.model.network.atom_attention"
PRODUCER = "atlasfold.model.network.rel_pos_encoding"
EXPECTED = ("disabled", "cpu", "cc<8")
_STATE = {"depth": 0, "override": None}          # override: None = read AFO_ATOM_TF32; True/False = forced in-process by bench_arm()


def enabled() -> bool:
    if _STATE["override"] is not None:
        return bool(_STATE["override"])
    return os.environ.get(ENV, "1") != "0"


def bench_arm(label: str) -> None:
    """In-process override of AFO_ATOM_TF32 (not a kit switch): label 'A' = the entry precision (lever inert), any other label = the TF32 window."""
    _STATE["override"] = (label != "A")


bench_arm.state = lambda: {"atom_tf32": enabled(), "depth": _STATE["depth"]}   # type: ignore[attr-defined]


@contextlib.contextmanager
def window(on: bool = True):
    """allow_tf32 := `on` for the body, the entry value restored on exit (also on error); depth-counted so the producer guard knows."""
    M = torch.backends.cuda.matmul
    prev = M.allow_tf32
    M.allow_tf32 = bool(on)
    _STATE["depth"] += 1 if on else 0
    try:
        yield prev
    finally:
        if on:
            _STATE["depth"] -= 1
        M.allow_tf32 = prev


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    from opt_core.counters import Ledger
    try:
        AA = importlib.import_module(TARGET)
        RP = importlib.import_module(PRODUCER)
    except Exception as e:  # noqa: BLE001
        return Installed("atom_tf32", False, reason=f"import:{type(e).__name__}")
    cc_ok = bool(torch.cuda.is_available()) and torch.cuda.get_device_capability(0)[0] >= 8
    ledger = Ledger(NAME, impl="cublas.allow_tf32(AtomAttentionStack.forward)", origin="kit", expected=EXPECTED)
    cls = AA.AtomAttentionStack
    stock = cls.forward

    def forward(self, batch, q, *args, **kwargs):
        if not enabled():
            ledger.fallback("disabled")
            return stock(self, batch, q, *args, **kwargs)
        if q.device.type != "cuda":
            ledger.fallback("cpu")
            return stock(self, batch, q, *args, **kwargs)
        if not cc_ok:
            ledger.fallback("cc<8")
            return stock(self, batch, q, *args, **kwargs)
        with window(True):
            out = stock(self, batch, q, *args, **kwargs)
        ledger.serve("capture" if torch.cuda.is_current_stream_capturing() else "eager")
        return out
    forward.__qualname__ = "AtomAttentionStack.forward[atlasfold_opt:atom_tf32]"
    rebind(cls, "forward", forward, stock)

    rp_cls = RP.AtomRelativePositionEncoding
    rp_stock = rp_cls.forward

    def rp_forward(self, *args, **kwargs):                       # KEEP_COORD guard: the reference-atom offset table look-up stays at the entry precision
        if _STATE["depth"] > 0 and torch.backends.cuda.matmul.allow_tf32:
            ledger.count("guard_relpos")
            with window(False):
                return rp_stock(self, *args, **kwargs)
        return rp_stock(self, *args, **kwargs)
    rp_forward.__qualname__ = "AtomRelativePositionEncoding.forward[atlasfold_opt:atom_tf32.guard]"
    rebind(rp_cls, "forward", rp_forward, rp_stock)
    return Installed("atom_tf32", True, lines=[lambda: ledger.line(tag)], gates=[size_gated(ledger)],
                     facts={"impl": getattr(ledger, "impl", None), "env": os.environ.get(ENV, "1"), "cc_ok": cc_ok, "ledger": ledger})
