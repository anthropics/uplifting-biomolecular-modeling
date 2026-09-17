"""The `token_agg` lever: the atom → token feature aggregation as ONE deterministic segment-reduce kernel (fast class).

Stock `aggregate_atom_feat_to_tokens(token_mask, atom_to_token_index, atom_mask, atom_feat, atom_dim, "mean")` (core/utils/atomize_utils.py;
the atom-attention encoder calls it once per denoiser step on the [.., S, N_atom, 768] atom activations) builds an int64 index tensor of the
feature's full shape per call (`.repeat` over the sample and channel dims — 461 MB at 2,024 tokens × 5 samples), scatter-adds the masked
features into [.., S, N_token + 1, C] with float atomics (run-to-run nondeterministic), scatter-adds the mask for the counts and divides.
The engine lays atoms out token by token (`atom_to_token_index` non-decreasing over the atom axis), so token j's atoms are one contiguous run:
this lever derives the runs once per rollout (`opt_core.kernels.dtk_kernels.segments`, memoised through `opt_core.of3_sampler.rollout_memo` — address-stable,
refreshed in place, so the captured step stays correct for a later input) and serves every call with `dtk_kernels.seg_reduce`: one read of
each feature row, fp32 accumulation in a fixed order, the loop bound a device scalar. Result = stock's arithmetic up to summation order
(≈1 ulp of an fp64 reference; tier 2 — deterministic where stock is not). Output dtype = stock's (promote(feat, mask)).

Domain: inference, batch size 1 (the index / masks' leading dims all 1), any sample count S, any token / atom / channel count, `mean` or
`sum` over atom_dim = -2, fp32 / bf16 / fp16 CUDA features. Anything else runs the stock function BY NAME (`fallback:<reason>` — atom_dim,
fn, dtype, batch_gt1, atom_count, unsorted_atoms); an unsorted layout met while a captured graph already contains the kernel fails the run
by name instead of replaying it wrong. Sites: `atomize_utils.aggregate_atom_feat_to_tokens` and the encoder module's own binding of that name
(`sequence_local_atom_attention.aggregate_atom_feat_to_tokens`), plus the rollout boundary (rollout_memo).

Switch (the kit adapter's name, `configure(ENV=…)`): <KIT>_TOKEN_AGG=seg_reduce; evidence `<PREFIX> installed …` and one exit line
`<PREFIX> LEVER name=token_agg state=on served=<n> fallback=<..> modes=<..>`. Engines: the OF3 code family (0.4.x and its 0.5.x fork); the
kits' `cells/token_agg.py` are the adapters (switch name, log prefix, the engine's module paths M_ATOMIZE / M_SLAA).
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict, Optional

from . import rollout_memo as RM

PREFIX = "[opt_core/of3_sampler.token_agg]"              # the kit adapter names itself and its switch: configure(PREFIX=, ENV=)
ENV: Optional[str] = None                                # <KIT>_TOKEN_AGG=seg_reduce
VALUES = ("seg_reduce",)
KERNEL = "dtk_kernels"                                            # the core's carried kernel module, gated by name at activation
M_ATOMIZE: Optional[str] = None                          # the engine module defining aggregate_atom_feat_to_tokens (….core.utils.atomize_utils)
M_SLAA: Optional[str] = None                             # the encoder module that imports it by name (….core.model.layers.sequence_local_atom_attention)
CONFIGURABLE = ("PREFIX", "ENV", "M_ATOMIZE", "M_SLAA")
def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words (log prefix, switch name, the engine's module paths) before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "impl": None, "served": 0, "fallback": {}, "modes": {}, "first": None,
                         "sorted": {}, "patched": [], "errors": {}}
_DK = {"mod": None}


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if ("token_agg", once_key) in RM._ONCE:
            return
        RM._ONCE.add(("token_agg", once_key))
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:                                                     # not bound by a kit adapter (configure(ENV=...)): nothing requested
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return STATE["state"] == "on"


def _refuse(reason: str) -> None:
    if STATE["state"] == "on":
        STATE["state"] = "refused"; STATE["reason"] = reason
        _log(f"REFUSED ({reason}): the stock aggregation runs")


def _make_agg(orig):
    def aggregate_atom_feat_to_tokens(token_mask, atom_to_token_index, atom_mask, atom_feat, atom_dim=-1, aggregate_fn="mean", eps=1e-9):
        stock = lambda: orig(token_mask, atom_to_token_index, atom_mask, atom_feat, atom_dim, aggregate_fn, eps)   # noqa: E731
        if not (serving() and RM.active()):
            return stock()
        import torch
        dk = _DK["mod"]
        why = None
        nd = atom_feat.dim()
        ad = atom_dim if atom_dim >= 0 else nd + atom_dim
        n_token = int(token_mask.shape[-1]); A = int(atom_mask.shape[-1])
        lead_b = 1
        for d_ in tuple(token_mask.shape[:-1]):
            lead_b *= int(d_)
        if ad != nd - 2:
            why = "atom_dim:%d/%d" % (atom_dim, nd)
        elif aggregate_fn not in ("mean", "sum"):
            why = "fn:%s" % aggregate_fn
        elif not atom_feat.is_cuda or atom_feat.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            why = "dtype:%s" % str(atom_feat.dtype).replace("torch.", "")
        elif lead_b != 1 or atom_to_token_index.numel() != A or atom_mask.numel() != A:
            why = "batch_gt1"
        elif int(atom_feat.shape[-2]) != A:
            why = "atom_count"
        if why is not None:
            _count(STATE["fallback"], why)
            _log("call feat=%s idx=%s -> the stock aggregation (fallback:%s)" % (tuple(atom_feat.shape), tuple(atom_to_token_index.shape), why), once_key=("fb", why))
            return stock()
        holder = {}

        def compute():
            st, ct, mx, ok = dk.segments(atom_to_token_index, atom_mask, n_token)
            holder["ok"] = ok
            return (st, ct, mx)
        st, ct, mx = RM.memo_call("segments", ("segments", RM.sig(atom_to_token_index), RM.sig(atom_mask), n_token), compute)
        if "ok" in holder:
            STATE["sorted"] = {"epoch": RM.epoch(), "ok": holder["ok"]}
        if STATE["sorted"].get("epoch") == RM.epoch() and not STATE["sorted"].get("ok", True):
            if RM.graphs_active() and STATE["served"] > 0:
                raise RuntimeError(f"{PREFIX} atom_to_token_index is not laid out token by token in this rollout while a captured graph already "
                                   "contains the segment kernel — refusing to replay it wrong; run with %s unset" % ENV)
            _count(STATE["fallback"], "unsorted_atoms")
            _log("atom_to_token_index is not laid out token by token in this rollout -> the stock aggregation", once_key="unsorted")
            return stock()
        C = int(atom_feat.shape[-1])
        try:
            f3 = atom_feat.reshape(-1, A, C)
            if f3.stride(-1) != 1:
                f3 = f3.contiguous(); _count(STATE["modes"], "feat_copy")
            mk = atom_mask.reshape(-1)
            if mk.dtype == torch.bool:
                mk = mk.to(torch.float32)
            if mk.stride(0) != 1:
                mk = mk.contiguous()
            out = dk.seg_reduce(f3, st, ct, mx, atom_mask=mk, mean=(aggregate_fn == "mean"), eps=eps,
                                out_dtype=torch.promote_types(atom_feat.dtype, atom_mask.dtype))     # stock: feat * mask promotes
        except Exception as e:  # noqa: BLE001
            from ..oom import is_oom
            if is_oom(e):
                raise
            if RM.graphs_active() and STATE["served"] > 0:              # a captured graph already holds the kernel: a Python-side fallback now would diverge from what replays
                raise RuntimeError(f"{PREFIX} the segment-reduce path failed ({type(e).__name__}: {e}) after a captured graph took the kernel — "
                                   "refusing to mix routes; run with %s unset" % ENV) from e
            why = "error:%s" % type(e).__name__
            _count(STATE["fallback"], why); _count(STATE["errors"], why)
            _log("served path failed: %r -> the stock aggregation for this call (fallback:%s)" % (e, why), once_key=("fb", why))
            return stock()
        STATE["served"] += 1
        _count(STATE["modes"], "%s:S=%d:%s" % (aggregate_fn, f3.shape[0], str(atom_feat.dtype).replace("torch.", "")))
        if STATE["first"] is None:
            STATE["first"] = "feat=%s:%s idx=%s n_token=%d" % (tuple(atom_feat.shape), str(atom_feat.dtype).replace("torch.", ""), tuple(atom_to_token_index.shape), n_token)
            _log("first served call " + STATE["first"])
        return out.reshape(*atom_feat.shape[:-2], n_token, C)
    aggregate_atom_feat_to_tokens._of3opt_token_agg = True; aggregate_atom_feat_to_tokens.__wrapped__ = orig
    return aggregate_atom_feat_to_tokens


def census_line() -> str:
    return (f"{PREFIX} LEVER name=token_agg state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" impl={STATE['impl']} served={STATE['served']} fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}"
            f" modes={','.join('%s:%d' % kv for kv in sorted(STATE['modes'].items())) or 'none'}"
            f" errors={','.join('%s:%d' % kv for kv in sorted(STATE['errors'].items())) or 'none'}" + (f" first={STATE['first']}" if STATE["first"] else ""))


def install(environ=None) -> dict:
    """Route the core kernel, patch both bindings of the stock function, arm the rollout boundary. Idempotent; raises by name when the core
    kernel is not importable."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not (M_ATOMIZE and M_SLAA):
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_ATOMIZE=, M_SLAA=) before install")
    from opt_core.kernels import route
    route(KERNEL)
    import importlib
    dk = importlib.import_module(KERNEL)
    if not (hasattr(dk, "seg_reduce") and hasattr(dk, "segments")):
        raise RuntimeError(f"{PREFIX} {KERNEL} at {getattr(dk, '__file__', None)} carries no seg_reduce / segments (opt_core >= 0.5.18.20)")
    _DK["mod"] = dk
    am = importlib.import_module(M_ATOMIZE); slaa = importlib.import_module(M_SLAA)
    orig = am.aggregate_atom_feat_to_tokens
    repl = _make_agg(orig) if not getattr(orig, "_of3opt_token_agg", False) else orig     # the occupant as found IS the named fallback (a foreign wrapper included)
    am.aggregate_atom_feat_to_tokens = repl; STATE["patched"].append(f"{M_ATOMIZE}.aggregate_atom_feat_to_tokens")
    cur = getattr(slaa, "aggregate_atom_feat_to_tokens", None)      # the encoder module imported the function by name: its global is a site too
    if cur is not None and not getattr(cur, "_of3opt_token_agg", False):
        slaa.aggregate_atom_feat_to_tokens = repl; STATE["patched"].append(f"{M_SLAA}.aggregate_atom_feat_to_tokens")
    STATE["state"] = "on"; STATE["impl"] = getattr(dk, "__file__", None)
    RM.register_user("token_agg", serving, _refuse)
    RM.arm_boundary()
    STATE["installed"] = True
    _log(f"installed: {' and '.join(STATE['patched'])} serve the atom->token mean with {KERNEL}.seg_reduce from {STATE['impl']} "
         "(runs derived once per rollout, rollout_memo)")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
