"""G1: a recurring fixed-shape scoring forward on the kit path replayed as ONE CUDA graph. Per (batch, length, device, dtype, inference-mode)
key the first two calls run eagerly — call 1 fills the shape's caches, plans and buffers, call 2 is where a GEMM-configuration lever settles
(kit/ltsel.py, when present; capture also waits for its `settled`) — so a length seen once or twice pays nothing here; call 3 issues its
eager forward and, while the device executes it, records the same forward into a graph on a side stream (a capture executes nothing: the
recording is host work overlapped with that forward) and returns the eager result; every later call copies the token ids into the captured
input buffer, replays, and hands the caller a fresh clone of the logits, never the graph's own output buffer. A replay issues the kernels
the eager call issued, on the same buffers, in the same order: the logits are the eager call's. Replay serves forwards of at most
TOKENS_MAX = 16,384 tokens (batch x length): what it removes is launch cadence, a small share of a larger forward, while the graph's private
pool holds one forward's intermediate tensors (reserved memory; the peak allocation is unchanged) and grows with the token count; a larger
shape runs eagerly, named once. Other calls the graph does not serve run eagerly, named once: a model on more than one device,
autograd recording, token ids not on the model's device, a call under stream capture, or a capture that raised (that key stays eager).
The graphs are shape-bound state: the shape manager releases them with its stores (a graph holds those buffers' addresses), so at most
the current shape's graph is resident. The per-forward tally counts every call once, replayed or not (it wraps this wrapper)."""
from __future__ import annotations

import sys

import torch

from evo2_opt.kit.base import CTR

try:                                                   # a GEMM-configuration lever that settles per shape over the first eager calls: capture waits for it
    from evo2_opt.kit import ltsel as _LT
except ImportError:
    _LT = None

LEVER = "G1_forward_graph_replay"
PREFIX = "[evo2-kit]"
TOKENS_MAX = 16384                 # replay serves forwards of at most this many tokens (batch x length); larger forwards run eagerly, named once
_graphs: dict = {}                 # key -> {"calls", "graph", "ids", "out"} ; "graph" None until captured, False when this key stays eager
_named: set = set()


def _name_once(tag: str, text: str) -> None:
    if tag not in _named:
        _named.add(tag)
        print(f"{PREFIX} G1: {text} — eager forward, named once; counted as g1_eager:{tag}", file=sys.stderr, flush=True)
    CTR[f"g1_eager:{tag}"] += 1


def _clone_out(out):
    if isinstance(out, torch.Tensor):
        return out.clone()
    if isinstance(out, tuple):
        return tuple(_clone_out(o) for o in out)
    if isinstance(out, list):
        return [_clone_out(o) for o in out]
    return out


def release() -> None:
    """Drop every graph (the shape manager calls this with its stores: the captured buffers are theirs)."""
    _graphs.clear()


def graphed(orig_forward, gate, devices):
    """Wrap vortex's StripedHyena.forward(self, x, inference_params_dict=None, padding_mask=None): graph replay on the kit path for scoring
    calls of a model on one device; everything else is the wrapped forward as it is."""
    one_device = len({str(d) for d in devices}) == 1

    def forward(self, x, inference_params_dict=None, padding_mask=None, *args, **kwargs):
        eager = lambda: orig_forward(self, x, inference_params_dict, padding_mask, *args, **kwargs)          # noqa: E731
        if gate.mode != "kit" or inference_params_dict is not None or padding_mask is not None or args or kwargs:
            return eager()
        if not one_device:
            _name_once("multi_device", "the model spans more than one device"); return eager()
        if torch.is_grad_enabled():
            _name_once("grad", "autograd is recording (no torch.no_grad / inference_mode around the call)"); return eager()
        if not (isinstance(x, torch.Tensor) and x.is_cuda and str(x.device) == str(devices[0])):
            _name_once("input_device", "the token ids are not a tensor on the model's device"); return eager()
        if torch.cuda.is_current_stream_capturing():
            return eager()
        if int(x.shape[0]) * int(x.shape[1]) > TOKENS_MAX:
            _name_once("tokens", f"a forward of more than {TOKENS_MAX:,} tokens (batch x length) is not replayed (first such shape {tuple(x.shape)})")
            return eager()
        key = (tuple(x.shape), x.dtype, str(x.device), torch.is_inference_mode_enabled())
        st = _graphs.get(key)
        if st is None:
            st = _graphs[key] = {"calls": 0, "graph": None, "ids": None, "out": None}
        st["calls"] += 1
        settled = getattr(_LT, "settled", lambda *a: True)(int(x.shape[0]), int(x.shape[1]), x.device)
        if st["calls"] <= 2 or st["graph"] is False or (st["graph"] is None and not settled):   # calls 1-2 of a key: the shape's caches fill, the GEMM
            CTR["g1_eager_call"] += 1                                                           # configuration settles; capture waits for both
            return eager()
        if st["graph"] is None:
            out = eager()                                                  # this call's result; the device executes it while the host records the graph
            try:
                with torch.inference_mode(False), torch.no_grad():
                    st["ids"] = x.clone()                                  # a plain (non-inference) buffer: writable under either mode later
                g = torch.cuda.CUDAGraph()
                cur = torch.cuda.current_stream(x.device)
                side = torch.cuda.Stream(device=x.device)
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    g.capture_begin()                                      # its own private pool: freed with the graph
                    try:
                        st["out"] = orig_forward(self, st["ids"], None, None)
                    finally:
                        g.capture_end()
                cur.wait_stream(side)
                st["graph"] = g
                CTR["g1_capture"] += 1
            except Exception as e:                                         # noqa: BLE001 — a capture the stack refuses: this key stays eager, named
                st["graph"] = False; st["ids"] = st["out"] = None
                torch.cuda.synchronize(x.device)
                _name_once("capture_failed", f"graph capture raised {type(e).__name__}: {str(e)[:160]}")
            return out
        st["ids"].copy_(x)
        st["graph"].replay()
        CTR["g1_replay"] += 1
        return _clone_out(st["out"])

    forward.__wrapped_stock__ = getattr(orig_forward, "__wrapped_stock__", orig_forward)
    forward.__g1__ = True
    return forward


def resident() -> dict:
    """{key: calls} of the keys seen and whether a graph is held — for tests and the tally."""
    return {str(k): {"calls": v["calls"], "graph": (v["graph"] is not None and v["graph"] is not False)} for k, v in _graphs.items()}
