"""The ``dtk`` lever: the diffusion transformer's token attention with pair bias
(``rf3.model.layers.af3_diffusion_transformer.AttentionPairBiasDiffusion``: softmax(Q·K/sqrt(c) + B) over j, times V, gated) served by
the shared core's Triton flash-attention-with-bias kernel ``opt_core.kernels.dtk_kernels.flash_bias_attn`` (routed by name and held to the
core's sums like the FPF kernels, ``stack.kernels_route``) instead of the two einsums that materialise the [I, I, H] logits.

Numerics class: tier 2 (online softmax in fp32 registers, P·V in the activations' dtype — the same class as SDPA; not bitwise with the
einsum path). It is a lever of the ``fast`` mode (modes.KIT_MODES ``kit_levers``).

Install seam: the same source rewrite the FPF add-on uses for its own ``dattn`` option — the stock attention block (the add-on's
``_DATTN_BLOCK`` text, the single copy of it in this tree) is replaced inside ``AttentionPairBiasDiffusion.forward`` by one call to
:func:`attn`, the method recompiled in the stock module's namespace and re-wrapped in foundry's ``activation_checkpointing``. ``dtk`` and
the add-on's ``dattn`` replace the same block: a row names at most one of them (refused by name otherwise).

Per call: a size gate (``opt_core.attn.size_gate``; ``MIN_I`` = 400 tokens — below it the stock block runs, the kernel is not faster
there) decides on I; a served call runs the kernel once per leading-batch sample (the diffusion batch), reading the pair bias re-laid to
a contiguous [H, I, J] copy in the activations' dtype (``BIAS_LAYOUT``); a bias whose leading shape matches neither one-per-batch nor shared is a counted fallback to the stock block (``fallback:bias_shape``),
never a silent one. Inside the sampler CUDA graph the Python side of a call runs at warm-up and capture only: the census counts those
calls, replays repeat the captured kernels.

Evidence: :func:`describe` (read by report.tally into ``dtk``) — on, impl (where the kernel executes from), bias layout, the gate's census
(calls = served + gated + fallback), ``ok`` and ``reason``: a fallback, or a seam no call reached, fails the run by name (fold.lever_failures).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

from . import _core

MIN_I = 400                                       # the size gate's lower bound in tokens (I); below it the stock block runs, counted `gated`
BIAS_LAYOUT = "relayout"                          # how the kernel reads the [I, J, H] pair bias: re-laid to a contiguous [H, I, J] copy per call
KERNEL = "dtk_kernels"                            # the routed core kernel module (opt_core.kernels.SUMS/dtk_kernels.json)
ADAPTER = "fpf_rf3_adapter"                       # the FPF add-on's module: carries _ORIG and the stock block text _DATTN_BLOCK
_NEW = """            A_I = _rf3opt_dtk_attn(Q_IH, K_IH, V_IH, B_IIH, G_IH, self.c)  # [rosettafold3_opt dtk] flash attention with pair bias
"""

STATE = {"on": False, "impl": None, "bias": None, "error": None, "conflict": None, "conflict_reason": None}
GATE = None                                       # opt_core.attn.size_gate.SizeGate, built by enable()


class DtkRefused(RuntimeError):
    """The lever cannot apply in this process (named precondition); stack.fpf_apply turns it into the NOT ACTIVE line."""


def make_gate(min_tokens: int = MIN_I):
    """The size gate on I (``opt_core.attn.size_gate``): calls below ``min_tokens`` run the stock block, counted ``gated``."""
    sg = _core.load("attn.size_gate")
    return sg.SizeGate(name="dtk", min_tokens=min_tokens)


def _stock_block(Q, K, V, B, G, c):
    """The stock attention block, verbatim arithmetic (af3_diffusion_transformer.py: Q/np.sqrt(c), einsum, softmax over j, einsum, gate)."""
    import numpy as np
    import torch
    Q = Q / np.sqrt(c)
    A_IIH = torch.softmax(torch.einsum("...ihd,...jhd->...ijh", Q, K) + B, dim=-2)
    A_I = torch.einsum("...ijh,...jhc->...ihc", A_IIH, V)
    return (G * A_I).flatten(start_dim=-2)


def attn(Q, K, V, B, G, c):
    """Q, K, V, G: [..., I, H, D]; B: [..., I, J, H] (pair bias, shared by the batch or one per sample); returns [..., I, H*D]."""
    import math
    import torch
    I_, H_, D_ = Q.shape[-3], Q.shape[-2], Q.shape[-1]
    if not GATE.decide(int(I_)).served:
        return _stock_block(Q, K, V, B, G, c)
    lead = tuple(Q.shape[:-3])
    n = 1
    for s in lead:
        n *= int(s)
    Qb, Kb, Vb, Gb = (t.reshape(n, I_, H_, D_) for t in (Q, K, V, G))
    nb = 1
    for s in tuple(B.shape[:-3]):
        nb *= int(s)
    if B.dim() < 3 or nb not in (1, n) or tuple(B.shape[-3:]) != (I_, I_, H_):   # neither shared nor one-per-sample: the stock block, counted
        GATE.fallback("bias_shape")
        return _stock_block(Q, K, V, B, G, c)
    Bb = B.reshape(nb, I_, I_, H_)
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else Vb.dtype
    if dt not in (torch.bfloat16, torch.float16):
        dt = Vb.dtype
    scale = 1.0 / math.sqrt(c)
    dk = sys.modules[KERNEL]
    out = None                                                              # one [n, I, H*D] buffer, each sample's gated product written into its slice (no per-call stack copy)
    bias_h = None
    for i in range(n):
        q = Qb[i].to(dt).permute(1, 0, 2)                                  # [H, I, D] views of [I, H, D]
        k = Kb[i].to(dt).permute(1, 0, 2)
        v = Vb[i].to(dt).permute(1, 0, 2)
        if bias_h is None or nb != 1:
            bm = Bb[0 if nb == 1 else i].permute(2, 0, 1)                  # [H, I, J]
            bias_h = bm.to(dt).contiguous()                                    # re-laid in the activations' dtype (16-bit under autocast, else theirs)
        o = dk.flash_bias_attn(q, k, v, bias=bias_h, out_dtype=dt, scale=scale)   # [I, H*D]: the routed kernel's contract is one [H, I, D] problem per call
        g = Gb[i] * o.view(I_, H_, D_)                                       # the stock gate product (its dtype: the promotion of G's and the kernel's)
        if out is None:
            out = torch.empty((n, I_, H_ * D_), dtype=g.dtype, device=g.device)
        out[i] = g.flatten(start_dim=-2)
    return out.reshape(*lead, I_, H_ * D_) if lead else out[0]


def enable(kernel_file: Optional[str] = None) -> dict:
    """Install the lever in this process (once): needs the FPF adapter imported (its ``_ORIG`` / ``_DATTN_BLOCK``), the routed kernel
    importable, and the add-on's own ``dattn`` off. Returns :func:`describe`. Raises :class:`DtkRefused` naming the failed precondition."""
    global GATE
    import importlib
    import inspect
    import textwrap
    if STATE["on"]:
        return describe()
    adp = sys.modules.get(ADAPTER)
    if adp is None or not hasattr(adp, "_DATTN_BLOCK") or not hasattr(adp, "_ORIG"):
        raise DtkRefused(f"{ADAPTER} is not imported (or lacks _DATTN_BLOCK/_ORIG): dtk installs through the FPF add-on's seam after apply_arm")
    if (getattr(adp, "DATTN", None) or {}).get("on"):
        raise DtkRefused("the add-on's dattn is on in this process: dtk and dattn replace the same attention block — a row names one of them")
    STATE["bias"] = BIAS_LAYOUT
    GATE = make_gate()
    try:
        dk = importlib.import_module(KERNEL)                               # the routed name (stack.kernels_route held it to the core's sums)
    except Exception as e:
        raise DtkRefused(f"{KERNEL} not importable: {type(e).__name__}: {e}") from e
    STATE["impl"] = getattr(dk, "__file__", None)
    if kernel_file and STATE["impl"] and os.path.realpath(STATE["impl"]) != os.path.realpath(kernel_file):
        raise DtkRefused(f"{KERNEL} executes from {STATE['impl']}, not the routed core copy {kernel_file}")
    import rf3.model.layers.af3_diffusion_transformer as DT
    from foundry.training.checkpoint import activation_checkpointing
    cls = DT.AttentionPairBiasDiffusion
    orig = adp._ORIG.setdefault("dattn_forward", cls.forward)             # the add-on's record of the stock method (shared with its dattn)
    inner = getattr(orig, "__wrapped__", None)
    if inner is None and getattr(orig, "__closure__", None):
        for cell in orig.__closure__:
            try:
                if inspect.isfunction(cell.cell_contents) and cell.cell_contents.__name__ == "forward":
                    inner = cell.cell_contents
                    break
            except ValueError:
                pass
    inner = inner or orig
    src = textwrap.dedent(inspect.getsource(inner))
    src = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("@"))
    blk = textwrap.dedent(adp._DATTN_BLOCK)
    blk8 = "\n".join(("        " + ln if ln else ln) for ln in blk.splitlines()) + "\n"
    new8 = "\n".join(("        " + ln if ln else ln) for ln in textwrap.dedent(_NEW).splitlines()) + "\n"
    if blk8 not in src:
        raise DtkRefused(f"the stock attention block ({ADAPTER}._DATTN_BLOCK) is not in {cls.__name__}.forward's source "
                         f"({inspect.getsourcefile(inner)}): the interpreter's rf3 differs from the shipped kit")
    ns = dict(DT.__dict__)
    ns["_rf3opt_dtk_attn"] = attn
    exec(compile(src.replace(blk8, new8), f"<rosettafold3_opt.dtk:{inspect.getsourcefile(inner)}>", "exec"), ns)   # noqa: S102
    cls.forward = activation_checkpointing(ns["forward"])
    STATE["on"] = True
    return describe()


def decline(conflict: str, reason: str) -> dict:
    """Record that the lever stays OFF BY NAME in this process (``conflict``: the one-token cause, e.g. ``n_gpu`` — the row-sharding
    adapter owns the token-attention statement under ``--n_gpu > 1``; ``reason``: the sentence). Nothing is installed; :func:`describe`
    carries the word so the exit tally, the LEVER line and ``pred``'s all-rank verdict name it. Returns :func:`describe`."""
    STATE["conflict"] = str(conflict)
    STATE["conflict_reason"] = str(reason)
    return describe()


def problems() -> list:
    """The fail-closed gate as sentences (empty = clean): any counted fallback; a seam that no call reached while the lever was on."""
    if not STATE["on"] or GATE is None:
        return []
    out = GATE.problems(allow_fallback=False)
    c = GATE.census()
    if c["calls"] == 0:
        out.append("no AttentionPairBiasDiffusion call reached the dtk seam in this process (the rewrite is installed but never ran)")
    return out


def describe() -> dict:
    """The exit tally's ``dtk`` block: on, impl, bias, min_tokens, the gate census, ok + reason."""
    census = GATE.census() if GATE is not None else None
    if not STATE["on"] and STATE["conflict"]:                              # off by name (decline): nothing to judge in this process
        return {"on": False, "impl": None, "origin": "core", "bias": None, "min_tokens": None, "census": None,
                "conflict": STATE["conflict"], "ok": True, "reason": f"off:conflict:{STATE['conflict']}: {STATE['conflict_reason']}"}
    bad = problems()
    return {"on": STATE["on"], "impl": STATE["impl"], "origin": "core", "bias": STATE["bias"],
            "min_tokens": (GATE.min_tokens if GATE is not None else None), "census": census,
            "ok": STATE["on"] and not bad, "reason": ("; ".join(bad) if bad else (None if STATE["on"] else "not installed"))}


def lever_evidence() -> list:
    """(key, value) pairs for the LEVER line after name/strategy/state: impl, origin, bias, then the gate's own pairs."""
    pairs = [("impl", STATE["impl"]), ("origin", "core"), ("bias", STATE["bias"])]
    if GATE is not None:
        pairs += GATE.lever_evidence()
    return pairs
