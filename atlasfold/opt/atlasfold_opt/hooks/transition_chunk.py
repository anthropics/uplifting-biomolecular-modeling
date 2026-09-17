"""Row-blocked pair Transition (LayerNorm -> SwiGLU -> Linear; primitives/transition.py) — two levers over ONE wrapper of Transition.forward.

Why: the stock Transition runs on the whole [B, L, L, c] pair tensor, so its SwiGLU materializes the Linear output [B, L, L, 2*hidden] plus the
[B, L, L, hidden] SiLU and product.  In the trunk (LMStack 4 blocks + Pairformer 48 blocks, block.py L281 `z = z + self.transition_z(z)`,
pair_transition_factor 4, bf16 under the runner's autocast) that transient is 8 + 4 + 4 times the bf16 pair tensor: the largest single allocation
of a prediction and the one that sets the peak / the token ceiling (run_lm_embedder -> trunk.py L161 -> block.py L281 -> activation.py L15, first
LM-stack block).  The diffusion PairConditioning's two Transitions (diffusion_transformer.py L137, factor 2, fp32) and the
confidence heads' pair blocks (bf16 under the runners' autocast — only the logits heads run fp32 — or fp32 when the model runs without autocast)
carry the same transient.  Every op of the Transition is position-wise (LayerNorm over
channels, two channel GEMMs, SiLU, a product): the same module applied to ROW BLOCKS of the pair tensor and written into one output is the same
arithmetic per element, and the row schedule is a pure function of the shape.

* `conf_transition_chunk`: calls made while a confidence head's forward is on the stack (ConfidenceHead_Monomer / _Multimer).
* `pair_transition_chunk`: every OTHER pair-shaped call — LM stack, main Pairformer stack, diffusion pair conditioning.  AFO_PAIR_TRANSITION_CHUNK=0
  switches it off (those calls then run stock and are counted on the confidence lever's line as `outside_conf_head`).
Both: only rank-4 square inputs whose SwiGLU Linear output would exceed 2 x AFO_TRANSITION_CHUNK_MIB (default 512 MiB per block) are blocked; smaller
or non-pair inputs run stock (expected fallbacks `small` / `not_pair`).  Class: exact where the row-blocked GEMMs equal the unblocked ones bitwise on
the pinned stack, fast otherwise (CHANGES.md records which, per lever)."""
from __future__ import annotations

import os
import threading

import torch

from opt_core.counters import Ledger

from . import Installed, size_gated

NAME = "LOCAL.atlasfold.conf_transition_chunk"
NAME_PAIR = "LOCAL.atlasfold.pair_transition_chunk"
_IN_HEAD = threading.local()
_STATE = {"conf": None, "pair": None, "stock": None, "TR": None}          # one wrapper, two ledgers


def chunk_rows(L: int, B: int, hidden2: int, elem: int, target_bytes: int) -> int:
    """Rows per block so that one block's SwiGLU Linear output [B, r, L, hidden2] stays <= target_bytes (>= 1 row)."""
    per_row = max(1, B * L * hidden2 * elem)
    return max(1, min(L, target_bytes // per_row))


def _target() -> int:
    return int(os.environ.get("AFO_TRANSITION_CHUNK_MIB", "512")) << 20


def _blocked(stock_tr, module, x, lever: dict):
    """The Transition on row blocks of x [B, L, L, c]; returns None when the call is out of scope for blocking (caller runs stock)."""
    ledger = lever["ledger"]
    if x.dim() != 4 or x.shape[-2] != x.shape[-3]:
        ledger.fallback("not_pair"); return None
    B, L = int(x.shape[0]), int(x.shape[-3])
    try:
        hidden2 = int(module[1].linear.weight.shape[0])                 # SwiGLU's Linear: channel -> 2 * hidden
    except Exception:  # noqa: BLE001
        ledger.fallback("not_pair"); return None
    dev = x.device.type
    elem = torch.get_autocast_dtype(dev).itemsize if torch.is_autocast_enabled(dev) else x.element_size()
    target = lever["target"]
    if B * L * L * hidden2 * elem <= 2 * target:
        ledger.fallback("small"); return None
    r = chunk_rows(L, B, hidden2, elem, target)
    n = -(-L // r)
    out = None
    for c in range(n):
        i0, i1 = c * r, min(L, (c + 1) * r)
        y = stock_tr(module, x[:, i0:i1])
        if out is None:
            out = torch.empty(x.shape[:-1] + (y.shape[-1],), dtype=y.dtype, device=y.device)
        out[:, i0:i1] = y
        del y
    lever["facts"]["max_chunks"] = max(lever["facts"]["max_chunks"], n)
    ledger.serve("L%dxC%d:r%dx%d" % (L, int(x.shape[-1]), r, n))
    return out


UPSTREAM_WORDS = ("upstream_transition_exact", "upstream_pair_transition")   # a transition KERNEL lever installed outside this wrapper served a pair-shaped call first (by name, per call)


def note_upstream(x, word: str) -> None:
    """Called by a transition kernel lever (`transition_exact`, fast's `pair_transition`: both install AFTER this wrapper, i.e. outside it) for
    each pair-shaped call ITS kernel served: that call never reaches the row-block wrapper, so the ledger that would have blocked it (the
    confidence-head ledger inside the heads, the pair ledger elsewhere) counts it as the named fallback ``word`` — the census then says who
    served the head's / trunk's pair Transitions instead of reading as if they were absent.  Never raises; non-pair shapes are ignored."""
    try:
        if word not in UPSTREAM_WORDS or x.dim() != 4 or x.shape[-2] != x.shape[-3]:
            return
        lever = _STATE["conf"] if getattr(_IN_HEAD, "depth", 0) else _STATE["pair"]
        if lever is not None:
            lever["ledger"].fallback(word)
    except Exception:  # noqa: BLE001
        pass


def _ensure_wrapped(TR):
    if _STATE["stock"] is not None:
        return
    stock_tr = TR.Transition.forward                                     # nn.Sequential.forward
    _STATE["stock"], _STATE["TR"] = stock_tr, TR

    def tr_forward(self, x):
        if getattr(_IN_HEAD, "depth", 0):
            lever = _STATE["conf"]
            if lever is None:
                return stock_tr(self, x)
            out = _blocked(stock_tr, self, x, lever)
            return stock_tr(self, x) if out is None else out
        pair = _STATE["pair"]
        if pair is None or os.environ.get("AFO_PAIR_TRANSITION_CHUNK", "1") == "0":
            if _STATE["conf"] is not None:
                _STATE["conf"]["ledger"].fallback("outside_conf_head")
            if pair is not None:
                pair["ledger"].fallback("disabled")
            return stock_tr(self, x)
        out = _blocked(stock_tr, self, x, pair)
        return stock_tr(self, x) if out is None else out

    tr_forward.__wrapped_stock__ = stock_tr
    TR.Transition.forward = tr_forward


def _maybe_unwrap():
    if _STATE["conf"] is None and _STATE["pair"] is None and _STATE["stock"] is not None:
        _STATE["TR"].Transition.forward = _STATE["stock"]
        _STATE["stock"], _STATE["TR"] = None, None


def _line_for(lever: dict, tag: str):
    def line():
        from opt_core import report as _r
        return lever["ledger"].line(tag) + " " + _r.kv(("chunks", lever["facts"]["max_chunks"]), ("block_mib", lever["target"] >> 20))
    return line


def install(mode: str, tag: str, settings: dict):
    """conf_transition_chunk: row blocks for the pair Transition inside the confidence heads."""
    try:
        from atlasfold.model.network.primitives import transition as TR
        from atlasfold.model.network import confidence_head as CH
    except Exception as e:  # noqa: BLE001
        return Installed("conf_transition_chunk", False, reason=f"import:{type(e).__name__}")
    lever = {"ledger": Ledger(NAME, impl="row-blocks(Transition)", origin="kit", expected=("outside_conf_head", "small", "not_pair") + UPSTREAM_WORDS),
             "target": _target(), "facts": {"max_chunks": 0}}
    _STATE["conf"] = lever
    _ensure_wrapped(TR)

    def head_wrapper(stock_head_forward, cls_name):
        def forward(self, *a, **k):
            _IN_HEAD.depth = getattr(_IN_HEAD, "depth", 0) + 1
            try:
                return stock_head_forward(self, *a, **k)
            finally:
                _IN_HEAD.depth -= 1
        forward.__wrapped_stock__ = stock_head_forward
        forward.__qualname__ = f"{cls_name}.forward[atlasfold_opt:conf_transition_chunk:scope]"
        return forward

    heads = {}
    for cls_name in ("ConfidenceHead_Monomer", "ConfidenceHead_Multimer"):
        cls = getattr(CH, cls_name, None)
        if cls is not None:
            heads[cls] = cls.forward
            cls.forward = head_wrapper(cls.forward, cls_name)

    def _restore():
        for cls, f in heads.items():
            cls.forward = f
        _STATE["conf"] = None
        _maybe_unwrap()
    facts = lever["facts"]; facts["target_mib"] = lever["target"] >> 20
    return Installed("conf_transition_chunk", True, lines=[_line_for(lever, tag)], gates=[size_gated(lever["ledger"])], facts={"restore": _restore, **facts})


def install_pair(mode: str, tag: str, settings: dict):
    """pair_transition_chunk: row blocks for every pair Transition OUTSIDE the confidence heads (LM stack, main stack, diffusion pair conditioning)."""
    try:
        from atlasfold.model.network.primitives import transition as TR
    except Exception as e:  # noqa: BLE001
        return Installed("pair_transition_chunk", False, reason=f"import:{type(e).__name__}")
    # AFO_PAIR_TRANSITION_CHUNK=0: the lever still installs (the mode row is served as listed) and every call runs stock, counted `disabled`
    lever = {"ledger": Ledger(NAME_PAIR, impl="row-blocks(Transition)", origin="kit", expected=("small", "not_pair", "disabled") + UPSTREAM_WORDS),
             "target": _target(), "facts": {"max_chunks": 0}}
    _STATE["pair"] = lever
    _ensure_wrapped(TR)

    def _restore():
        _STATE["pair"] = None
        _maybe_unwrap()
    facts = lever["facts"]; facts["target_mib"] = lever["target"] >> 20
    return Installed("pair_transition_chunk", True, lines=[_line_for(lever, tag)], gates=[size_gated(lever["ledger"])], facts={"restore": _restore, **facts})
