"""The KERNELS census — what the pair stack's kernel-servable operations ran on in one driver process, in the release's one census grammar
(``[genie3-opt] KERNELS route=g3batch trimul=<word> triatt=<word> cueq=<word> deepspeed=<word> tf32_matmul=<0|1>``), printed by the
batched driver at exit (g3batch.py) and recorded as ``timings.kernels``.

Genie 3's denoiser (structure_net IPA + a five-block pair transform of TriangleMultiplication out/in and a ReLU PairTransition,
latent/transformer.py) imports no vendor kernel library: there is no triangle ATTENTION in the architecture and upstream never imports
cuEquivariance, DeepSpeed/DS4Sci or flash-attn (``src/genie3`` has no such import) — those words are constant facts of the checkout
(``n/a-upstream:…``), stated so the line keeps the shared core's fixed census columns. The one servable operation is the
TriangleMultiplicativeUpdate (10 calls per denoiser call): word ``trimul`` = ``off-by-route:stock`` (the exact line: the module's own forward)
| ``engaged:fpf_trimul_v4@<version>-<origin>[served=<n>,fallback=0]`` / ``partial:…`` (lever L7, mode fast's provider:
the shared core's fused kernel served the eager warm-up + capture calls and its gate holds; `partial` counts the calls under the kernel's token floor
that ran the module's forward) | ``declined:below_min_tokens[…]`` (every call under the floor: the lever declined the request by name, the module's
forward ran) | ``fallback:<gate reason>[…]`` (the gate refused: an unexpected fallback reason or a kernel error — lever L7 then has no evidence and the
pass is partial). ``EXPECT`` is what a line must print for its TriangleMultiplication provider word (modes.trimul_of: stock | fpf); ``verdict``
compares a census to it.
"""
from __future__ import annotations

from typing import Optional

from .codes import TAG

ROUTE = "g3batch"
CONSTANT_WORDS = {
    "triatt": "n/a-upstream:no-triangle-attention-in-architecture",
    "cueq": "n/a-upstream:not-imported",
    "deepspeed": "n/a-upstream:not-imported",
}
EXPECT = {                                   # the line's TriangleMultiplication provider word (modes.trimul_of) -> the trimul census word's required prefix
    "stock": ("off-by-route:stock",),        # every exact line
    "fpf": ("engaged:fpf_trimul_v4@", "partial:fpf_trimul_v4@", "declined:below_min_tokens["),   # fast (lever L7): the kernel served (every call | all but the EXPECTED counted stock fallbacks) and its gate holds | nothing to serve (every call under the floor): declined by name
}
KERNELS_REPLACED = {"fpf": ("TriangleMultiplicativeUpdate.forward x10 per denoiser call -> opt_core fpf_trimul_v4 generic entry (F2.fpf_trimul_fast)",), "stock": ()}


def census(trimul_evidence: Optional[dict], tf32: Optional[bool]) -> dict:
    """The census record: {route, trimul (word), triatt, cueq, deepspeed, tf32_matmul, served, fallback, errors, gate_ok}."""
    from . import trimul as KT
    ev = trimul_evidence or {"mode": "stock"}
    word = ev.get("word") or KT.word(ev)
    c = ev.get("census") or {}
    return {"route": ROUTE, "trimul": word, **CONSTANT_WORDS, "tf32_matmul": None if tf32 is None else int(bool(tf32)),
            "served": int(c.get("served") or 0), "fallback": dict(c.get("fallback") or {}), "errors": dict(c.get("errors") or {}),
            "gate_ok": bool((ev.get("gate") or {"ok": True}).get("ok", True))}


def line(c: dict) -> str:
    """``[genie3-opt] KERNELS route=g3batch trimul=… triatt=… cueq=… deepspeed=… tf32_matmul=…``"""
    return (f"[{TAG}] KERNELS route={c.get('route', ROUTE)} trimul={c.get('trimul')} triatt={c.get('triatt')} cueq={c.get('cueq')} "
            f"deepspeed={c.get('deepspeed')} tf32_matmul={c.get('tf32_matmul')}")


def verdict(c: dict, trimul: str) -> str:
    """``ok`` when the census's trimul word is what a line with this TriangleMultiplication provider must print (EXPECT), else ``FAIL: <why>``."""
    want = EXPECT.get(trimul)
    if want is None:
        return f"FAIL: no expectation for trimul provider {trimul!r}"
    got = str(c.get("trimul") or "")
    if not got.startswith(tuple(want)):
        return f"FAIL: a --trimul {trimul} line expects trimul={'|'.join(want)}… but the process printed trimul={got}"
    if trimul == "fpf" and int(c.get("served") or 0) < 1 and not got.startswith("declined:"):
        return "FAIL: --trimul fpf expects the fused TriMul kernel to have served >= 1 call (served=0)"
    return "ok"
