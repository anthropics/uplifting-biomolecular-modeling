"""The `atom_window` lever: the diffusion sampler's sequence-local atom attention on two fused kernels (fast class).

Upstream's atom-attention encoder and decoder (AtomAttentionEncoder / AtomAttentionDecoder: 3 + 3 DiffusionTransformerBlocks with
`use_cross_attention`, 200 denoiser steps over the sample batch) run each block's attention half as ~40 kernels per step: the 128-key windows of
every 32-query block are GATHERED into [S, NB, 128, C] copies, the key-side AdaLN and the k / v projections run on those copies (every atom ~4x),
the [S, NB, H, 32, 128] logits are materialised three times (bias add, mask add, softmax) and the AdaLN-Zero gate + residual are separate passes.
This lever serves the same mathematics from the core's carried `atom_window` kernels (opt_core.kernels.atom_window, ATOM_WINDOW v0.1):

    ln_qkvg      LayerNorm + both AdaLN modulations with per-ATOM conditioning rows (never expanded over the samples) + ONE pass of the four
                 C x C projections -> [ q | k | v | sigmoid(g) ] column slices, k / v once per atom;
    window_attn  per (query block, sample): the shifted key window read IN PLACE (window_starts: upstream's get_block_indices rule as one start
                 index per block), q k^T + pair bias + upstream's key / pair validity as -inf, fp32 softmax, p v, sigmoid gate, W_o, the
                 AdaLN-Zero gate and the residual — one kernel;

then upstream's own conditioned transition (the block's second residual branch) unchanged. fp32 in / fp32 out; the dots are TF32 tensor-core dots
with round-to-nearest operands (precision word `tf32rn`: the numerics class of upstream's cuBLAS TF32 GEMMs in this fp32 island under
float32_matmul_precision "high"), or `tf32x3` / `ieee` on request. Not bitwise: outputs stay within the fast line's tolerance class.

Step-invariant operands. Per block and rollout the conditioning-side tensors (sigmoid(linear_g(LN c_l)) / linear_s(LN c_l) of both AdaLNs, the
AdaLN-Zero gate sigmoid(linear_ada_out(c_l)), the pair bias linear_z(LN_z(p_lm)) permuted to [NB, H, 32, 128]) and the mask-side ones (atom mask
rows, the real-atom count, the window starts) depend only on the rollout's inputs: they are computed with upstream's own modules at the first
denoiser step of each rollout and memoised in the address-stable per-rollout store (opt_core.of3_sampler.rollout_memo: refreshed in place at the eager first
step of every rollout under the CUDA-graphed step, dropped at the rollout's exit on the eager route; the store's byte cap applies). With
<KIT>_ATOM_WINDOW_INV=hoist the conditioning-side tensors are READ from the `atom_hoist` lever's published `_of3opt_atom_inv` buffers of the
same rollout epoch instead (default `own`: computed here — ~0.2 ms per block per rollout, no coupling).

Composition. DiffusionTransformerBlock.forward is patched class-wide and dispatches per call by instance: blocks with `use_cross_attention` inside a
rollout (rollout_memo.active()) take this schedule; every other call — the token diffusion transformer's blocks (served by `dit_glue` when that
lever is on: the two patches nest in either order, each delegating the other's blocks to the method it wrapped), the input embedder's atom
encoder outside the sampler, training / grad mode — goes to the wrapped method unchanged. The kernels are compiled once per module geometry when a
DiffusionModule is constructed (outside any forward).

Domain (refused BY NAME -> the wrapped block for that call, counted in the census): fp32 CUDA activations; batch 1 (leading dims of size 1); the
conditioning `s` without a sample dimension ([.., A, C]); the pair input z = LN_z(p_lm) as [.., NB, n_query, n_key, c_z]; an atom mask [.., S|1, A]
with the same real-atom count in every sample; C = H * D <= 128 with D in {16, 32, 64}; (n_query, n_key) = (32, 128); no
unknown keyword arguments. Anything unexpected inside the fused work is named, counted, and the wrapped block runs on the untouched inputs.

Switches (the kit adapter's names, `configure(ENV=, ENV_PRECISION=, ENV_INV=)`; read once, at install; the fast line exports the first):
    <KIT>_ATOM_WINDOW=1                       on
    <KIT>_ATOM_WINDOW_PRECISION=tf32rn        tf32rn (default) | tf32x3 | ieee — the kernels' dot precision word
    <KIT>_ATOM_WINDOW_INV=own                 own (default) | hoist — where the conditioning-side invariants come from

Census (stderr at exit, one line): <PREFIX> LEVER name=atom_window state=on|off ... calls= served= outside= fallback=<reason:n,> errors= inv=
precision= warmup_s= + the rollout store's words (rollout_memo.fields()). Engines: the OF3 code family (0.4.x and its 0.5.x fork); the kits'
`cells/atom_window.py` are the adapters (switch names, log prefix, the engine's module paths M_DIT / M_DM).
"""
from __future__ import annotations

import atexit
import math
import os
import sys
from typing import Any, Dict, Optional

from . import rollout_memo as RM

PREFIX = "[opt_core/of3_sampler.atom_window]"          # the kit adapter names itself and its switches: configure(PREFIX=, ENV=, ENV_PRECISION=, ENV_INV=)
ENV: Optional[str] = None                                # <KIT>_ATOM_WINDOW=1
ENV_PRECISION: Optional[str] = None                      # <KIT>_ATOM_WINDOW_PRECISION=tf32rn|tf32x3|ieee
ENV_INV: Optional[str] = None                            # <KIT>_ATOM_WINDOW_INV=own|hoist
VALUES = ("1",)
PRECISION_VALUES = ("tf32rn", "tf32x3", "ieee")
INV_VALUES = ("own", "hoist")
KERNEL = "atom_window"                                   # the core's carried kernel module, gated by name at activation
M_DIT: Optional[str] = None                              # the engine module defining DiffusionTransformerBlock (….core.model.layers.diffusion_transformer)
M_DM: Optional[str] = None                               # the engine module defining DiffusionModule (….core.model.structure.diffusion_module)
CONFIGURABLE = ("PREFIX", "ENV", "ENV_PRECISION", "ENV_INV", "M_DIT", "M_DM")


def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words (log prefix, switch names, the engine's module paths) before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


HOIST_ATTR = "_of3opt_atom_inv"                                   # atom_hoist's per-block published buffers {epoch, bias, q_gate, q_shift, k_gate, k_shift, ada_gate, ...}
HOIST_KEYS = {"bias": "bias", "gq": "q_gate", "lsq": "q_shift", "gk": "k_gate", "lsk": "k_shift", "gate": "ada_gate"}
NAME_ATTR = "_of3opt_atom_window_name"                            # enc.k / dec.k on the sampler's atom blocks (census / log words)
MARK = "_of3opt_atom_window"
STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "impl": None, "precision": "tf32rn", "inv": "own",
                         "calls": 0, "served": 0, "outside": 0, "fallback": {}, "errors": {}, "inv_source": {}, "warmup_s": None, "warmups": 0,
                         "blocks_named": 0, "patched": []}
_AW = {"mod": None}
_ONCE = set()
_GEOMS = set()
_RAGGED = {"key": None}                                            # (rollout epoch, mask signature) of a mask refused as ragged_samples                                                    # block geometries whose kernels are compiled in this process
SERVED_WINDOW = (32, 128)                                          # (n_query, n_key) the lever serves — the geometry the addressing and numerics tests cover
FLAGS = ("use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_triton_triangle_kernels", "use_lma", "use_high_precision_attention")


class Refuse(Exception):
    """A by-name refusal raised before any fused work touched the output: the wrapped block serves the call."""


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if once_key in _ONCE:
            return
        _ONCE.add(once_key)
    sys.stderr.write(f"{PREFIX} {msg}\n"); sys.stderr.flush()


def _count(d: dict, k, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def requested(environ=None) -> bool:
    """<KIT>_ATOM_WINDOW (ENV): unset/empty -> False; "1" -> True (the precision / inv words validated too); anything else -> ValueError."""
    environ = os.environ if environ is None else environ
    if ENV is None:
        raise RuntimeError(f"{__name__}: configure(ENV=...) first (the kit adapter binds the switch name)")
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r}: expected one of {', '.join(VALUES)} or unset")
    p = ((environ.get(ENV_PRECISION) if ENV_PRECISION else None) or "tf32rn").strip()
    if p not in PRECISION_VALUES:
        raise ValueError(f"{ENV_PRECISION}={p!r}: expected one of {', '.join(PRECISION_VALUES)} or unset")
    inv = ((environ.get(ENV_INV) if ENV_INV else None) or "own").strip()
    if inv not in INV_VALUES:
        raise ValueError(f"{ENV_INV}={inv!r}: expected one of {', '.join(INV_VALUES)} or unset")
    return True


def serving() -> bool:
    return STATE["state"] == "on"


def _refuse_lever(reason: str) -> None:
    """Turn the lever off by name for the rest of the process (rollout_memo's refusal: the per-rollout refresh cannot be placed)."""
    if STATE["state"] == "on":
        STATE["state"] = "off"; STATE["reason"] = reason
        _log(f"REFUSED ({reason}): every atom block takes the wrapped path from here")


# ------------------------------------------------------------------------------------------------------------------ the plan (domain)
def plan(blk, a, s, z, mask, kw: dict):
    """(True, None) when this call is inside the lever's domain, else (False, reason) — the reason is the census word."""
    import torch
    unknown = [k for k in kw if k not in FLAGS and k != "_mask_trans"]
    if unknown:
        return False, "kwargs:" + ",".join(sorted(unknown))
    apb = getattr(blk, "attention_pair_bias", None)
    if apb is None or not hasattr(apb, "layer_norm_a_k") or not hasattr(apb, "n_query"):
        return False, "module_layout"
    if not a.is_cuda:
        return False, "device_cpu"
    if a.dtype != torch.float32:
        return False, "dtype_%s" % str(a.dtype).replace("torch.", "")
    if s is None or mask is None or z is None:
        return False, "no_s_z_or_mask"
    if a.dim() < 3:
        return False, "a_rank"
    A_, C = int(a.shape[-2]), int(a.shape[-1])
    if any(int(d) != 1 for d in a.shape[:-3]):
        return False, "lead_dims"
    if tuple(s.shape[-2:]) != (A_, C) or s.numel() != A_ * C:
        return False, "s_has_sample_dim"
    if int(mask.shape[-1]) != A_ or any(int(d) != 1 for d in mask.shape[:-2]) or (mask.dim() >= 2 and int(mask.shape[-2]) not in (1, int(a.shape[-3]))):
        return False, "mask_shape"
    H = int(apb.mha.no_heads); D = C // max(H, 1)
    if D * H != C or D not in (16, 32, 64) or C % 16 or C > 128:                 # the kernels hold a C x C weight tile in shared memory: C <= 128
        return False, "head_geom"
    nq, nk = int(apb.n_query), int(apb.n_key)
    if (nq, nk) != SERVED_WINDOW:                                               # the tested geometry; other (multiple-of-16, power-of-two) pairs compile but are not served
        return False, "window_geom"
    if tuple(z.shape[-4:-1]) != (math.ceil(A_ / nq), nq, nk) or any(int(d) != 1 for d in z.shape[:-4]):
        return False, "z_shape"
    if z.shape[-1] != apb.linear_z.in_features:
        return False, "z_channels"
    return True, None


# ------------------------------------------------------------------------------------------------------------------ invariants
def _hoist_members(blk):
    """{my key: tensor} for the conditioning-side members the atom_hoist lever publishes on this block for the CURRENT rollout epoch
    (ENV_INV=hoist), else {} (absent, or another epoch's: counted `hoist_stale`)."""
    pub = blk.attention_pair_bias.__dict__.get(HOIST_ATTR)
    if not pub:
        return {}
    ep = pub.get("epoch") if isinstance(pub, dict) else getattr(pub, "epoch", None)
    if ep is not None and ep != RM.epoch():
        _count(STATE["inv_source"], "hoist_stale")
        return {}
    get = pub.get if isinstance(pub, dict) else (lambda k, d=None: getattr(pub, k, d))
    out = {}
    for mine, theirs in HOIST_KEYS.items():
        t = get(theirs)
        if t is not None:
            out[mine] = t if t.is_contiguous() else t.contiguous()      # the kernels index these densely as [A, C] / [NB, H, NQ, NK]
    return out


def invariants(blk, s, z, mask) -> dict:
    """The block's step-invariant kernel operands for this rollout: bias [NB, H, NQ, NK], gq / lsq / gk / lsk / gate [A, C], amask [S|1, A],
    n_real int32 [1], ks int32 [NB] — from the per-rollout store (computed with upstream's modules at the rollout's first step), the
    conditioning-side ones from atom_hoist's published buffers when STATE['inv'] == 'hoist' and they are current."""
    import torch
    AW = _AW["mod"]
    apb = blk.attention_pair_bias
    A_ = int(mask.shape[-1]); nq, nk, H = int(apb.n_query), int(apb.n_key), int(apb.mha.no_heads)
    out = _hoist_members(blk) if STATE["inv"] == "hoist" else {}
    if out:
        _count(STATE["inv_source"], "hoist")

    def cond():
        C = s.shape[-1]
        s2 = s.reshape(A_, C)
        lq, lk = apb.layer_norm_a_q, apb.layer_norm_a_k
        sq = lq.layer_norm_s(s2); sk = lk.layer_norm_s(s2)
        return (torch.sigmoid(lq.linear_g(sq)).contiguous(), lq.linear_s(sq).contiguous(), torch.sigmoid(lk.linear_g(sk)).contiguous(),
                lk.linear_s(sk).contiguous(), torch.sigmoid(apb.linear_ada_out(s2)).contiguous())

    def bias():
        return (apb.linear_z(z).reshape(-1, nq, nk, H).permute(0, 3, 1, 2).contiguous(),)

    rkey = (RM.epoch(), RM.sig(mask))
    if _RAGGED.get("key") == rkey:                                              # this rollout's mask was found ragged at its eager first step: refuse without a second sync
        raise Refuse("ragged_samples")

    def maskside():
        am = mask.reshape(-1, A_).to(torch.float32).contiguous()
        n_real_rows = am.sum(-1)
        if am.shape[0] > 1 and not bool((n_real_rows == n_real_rows[0]).all()):   # one device sync per rollout (the memo's eager first step)
            _RAGGED["key"] = rkey
            raise Refuse("ragged_samples")
        return (am, n_real_rows[:1].round().to(torch.int32), AW.window_starts(A_, n_real_rows[0], nq, nk, mask.device))

    need_cond = [k for k in ("gq", "lsq", "gk", "lsk", "gate") if k not in out]
    if need_cond:
        vals = RM.memo_call("atom_window.cond", ("atom_window.cond", id(blk), RM.sig(s)), cond)
        got = dict(zip(("gq", "lsq", "gk", "lsk", "gate"), vals))
        out.update({k: got[k] for k in need_cond})
        _count(STATE["inv_source"], "own")
    if "bias" not in out:
        out["bias"] = RM.memo_call("atom_window.bias", ("atom_window.bias", id(blk), RM.sig(z)), bias)[0]
    out["amask"], out["n_real"], out["ks"] = RM.memo_call("atom_window.mask", ("atom_window.mask", RM.sig(mask), nq, nk), maskside)
    return out


# ------------------------------------------------------------------------------------------------------------------ the fused block
def block_forward(blk, a, s, z, mask, mask_trans: bool = True):
    """DiffusionTransformerBlock.forward for an atom block inside the domain: a + gate * W_o(windowed pair-bias attention) in two kernels, then
    upstream's conditioned transition. Returns the block output (same shape / dtype as `a`)."""
    AW = _AW["mod"]
    inv = invariants(blk, s, z, mask)
    apb = blk.attention_pair_bias; mha = apb.mha
    A_, C = int(a.shape[-2]), int(a.shape[-1])
    a3 = a.reshape(-1, A_, C)
    if a3.stride(-1) != 1:
        a3 = a3.contiguous()
    H = int(mha.no_heads)
    prec = STATE["precision"]
    qkvg = AW.ln_qkvg(a3, inv["gq"], inv["lsq"], inv["gk"], inv["lsk"], mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g,
                      apb.layer_norm_a_q.eps, 1.0 / math.sqrt(C // H), precision=prec)
    out = AW.window_attn(qkvg, a3, inv["bias"], inv["ks"], inv["n_real"], inv["amask"], inv["gate"], mha.linear_o.weight, mha.linear_o.bias, H,
                         n_query=int(apb.n_query), n_key=int(apb.n_key), inf=float(apb.inf), precision=prec)
    out = out.view(a.shape)
    return out + blk.conditioned_transition(a=out, s=s, mask=mask if mask_trans else None)


def _patch_block(mod) -> bool:
    Blk = mod.DiffusionTransformerBlock
    if getattr(Blk.forward, MARK, False):
        return False
    orig = Blk.forward
    import inspect
    has_mask_trans = "_mask_trans" in inspect.signature(orig).parameters

    def forward(self, a, s, z, mask=None, **kw):
        wrapped = lambda: orig(self, a=a, s=s, z=z, mask=mask, **kw)   # noqa: E731
        if STATE["state"] != "on" or not getattr(self, "use_cross_attention", False):
            return wrapped()
        STATE["calls"] += 1
        if not RM.active():                                            # outside a rollout (the input embedder's atom encoder) or grad mode: upstream's block
            STATE["outside"] += 1
            return wrapped()
        mask_trans = bool(kw.get("_mask_trans", True)) if has_mask_trans else True
        try:
            ok, why = plan(self, a, s, z, mask, kw)
        except Exception as e:  # noqa: BLE001 — a plan that cannot be made is a named fallback; an out-of-memory is the caller's
            from ..oom import is_oom
            if is_oom(e):
                raise
            ok, why = False, "plan_error:%s" % type(e).__name__
        if not ok:
            _count(STATE["fallback"], why)
            _log("block call a=%s -> the wrapped block (fallback:%s)" % (tuple(a.shape), why), once_key=("fb", why))
            return wrapped()
        try:
            out = block_forward(self, a, s, z, mask, mask_trans=mask_trans)
        except Refuse as e:                                                            # a by-name refusal before the fused work: the wrapped block, counted
            why = str(e) or "refused"
            _count(STATE["fallback"], why)
            _log("block call a=%s -> the wrapped block (fallback:%s)" % (tuple(a.shape), why), once_key=("fb", why))
            return wrapped()
        except Exception as e:  # noqa: BLE001 — anything unexpected: named, counted, the wrapped block runs (inputs untouched); an out-of-memory is the caller's
            from ..oom import is_oom
            if is_oom(e):
                raise
            why = "error:%s" % type(e).__name__
            _count(STATE["errors"], why)
            _log("block call a=%s raised %r -> the wrapped block (counted %s)" % (tuple(a.shape), e, why), once_key=("err", why))
            return wrapped()
        if STATE["served"] == 0:
            _log("first fused atom block (%s): a=%s precision=%s inv=%s" % (self.__dict__.get(NAME_ATTR, "?"), tuple(a.shape), STATE["precision"], STATE["inv"]))
        STATE["served"] += 1
        return out

    setattr(forward, MARK, True); forward.__wrapped__ = orig
    Blk.forward = forward
    STATE["patched"].append(f"{M_DIT}.DiffusionTransformerBlock.forward")
    return True


# ------------------------------------------------------------------------------------------------------------------ construction-time naming + JIT warm-up
def _atom_blocks(dm):
    """[(name, block)] for the sampler DiffusionModule's atom-attention encoder / decoder blocks."""
    out = []
    for label, sub in (("enc", getattr(dm, "atom_attn_enc", None)), ("dec", getattr(dm, "atom_attn_dec", None))):
        at = getattr(sub, "atom_transformer", None) if sub is not None else None
        tf = getattr(at, "diffusion_transformer", at)
        for k, blk in enumerate(getattr(tf, "blocks", []) or []):
            out.append(("%s.%d" % (label, k), blk))
    return out


def _warm(blocks) -> None:
    """Compile the kernels for every distinct block geometry (C, H, n_query, n_key, inf, bias presence) on a dummy problem, once per process."""
    import time

    import torch
    if not torch.cuda.is_available():
        return
    AW = _AW["mod"]
    seen = _GEOMS
    t0 = time.perf_counter(); n = 0
    for _, blk in blocks:
        apb = getattr(blk, "attention_pair_bias", None)
        if apb is None:
            continue
        mha = apb.mha
        C = mha.linear_q.in_features; H = int(mha.no_heads)
        has_bias = tuple(l.bias is not None for l in (mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g, mha.linear_o))
        geom = (C, H, int(apb.n_query), int(apb.n_key), float(apb.inf), has_bias, STATE["precision"])
        if geom in seen or (C // H) * H != C or (C // H) not in (16, 32, 64) or C % 16 or C > 128 or (int(apb.n_query), int(apb.n_key)) != SERVED_WINDOW:
            continue
        seen.add(geom)
        with torch.no_grad():
            AW.warmup(C, H, int(apb.n_query), int(apb.n_key), float(apb.inf), has_bias, precision=STATE["precision"], device="cuda")
        n += 1
    if n:
        dt = time.perf_counter() - t0
        STATE["warmup_s"] = round((STATE["warmup_s"] or 0.0) + dt, 2); STATE["warmups"] += n
        _log("kernels compiled for %d block geometr%s in %.1f s at model construction (outside the forward)" % (n, "y" if n == 1 else "ies", dt))


def _patch_dm_init(mod) -> bool:
    DM = getattr(mod, "DiffusionModule", None)
    if DM is None or getattr(DM.__init__, MARK, False):
        return False
    orig = DM.__init__

    def __init__(self, *a, **kw):
        orig(self, *a, **kw)
        if STATE["state"] != "on":
            return
        try:
            blocks = _atom_blocks(self)
            for name, blk in blocks:
                blk.__dict__[NAME_ATTR] = name
            STATE["blocks_named"] += len(blocks)
            _warm(blocks)
        except Exception as e:  # noqa: BLE001 — naming / warm-up are conveniences: a failure is counted, the lever still serves (JIT then lands in the first step); an out-of-memory is the caller's
            from ..oom import is_oom
            if is_oom(e):
                raise
            _count(STATE["errors"], "construct:%s" % type(e).__name__)
            _log("construction hook: %r (counted; the kernels compile at first use instead)" % (e,))

    setattr(__init__, MARK, True); __init__.__wrapped__ = orig
    DM.__init__ = __init__
    STATE["patched"].append(f"{M_DM}.DiffusionModule.__init__")
    return True


# ------------------------------------------------------------------------------------------------------------------ install / census
def census_line() -> str:
    fb = ",".join("%s:%d" % kv for kv in sorted(STATE["fallback"].items()) if kv[1]) or "none"
    er = ",".join("%s:%d" % kv for kv in sorted(STATE["errors"].items())) or "none"
    src = ",".join("%s:%d" % kv for kv in sorted(STATE["inv_source"].items()) if kv[1]) or "none"
    return (f"{PREFIX} LEVER name=atom_window state={STATE['state']}{(' reason=' + STATE['reason']) if STATE['reason'] else ''} "
            f"precision={STATE['precision']} inv={STATE['inv']} inv_source={src} impl={STATE['impl']} calls={STATE['calls']} served={STATE['served']} "
            f"outside={STATE['outside']} fallback={fb} errors={er} warmup_s={STATE['warmup_s']} blocks_named={STATE['blocks_named']} {RM.fields()}")


def install(environ=None) -> dict:
    """Route the core kernel, patch DiffusionTransformerBlock.forward class-wide and DiffusionModule.__init__, arm the rollout boundary.
    Idempotent; raises by name when the core kernel is not importable."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if M_DIT is None or M_DM is None:
        raise RuntimeError(f"{PREFIX} configure(M_DIT=..., M_DM=...) first (the kit adapter binds the engine's module paths)")
    STATE["precision"] = ((environ.get(ENV_PRECISION) if ENV_PRECISION else None) or "tf32rn").strip()
    STATE["inv"] = ((environ.get(ENV_INV) if ENV_INV else None) or "own").strip()
    from opt_core.kernels import route
    route(KERNEL)
    import importlib
    aw = importlib.import_module(KERNEL)
    for fn in ("ln_qkvg", "window_attn", "window_starts", "warmup"):
        if not hasattr(aw, fn):
            raise RuntimeError(f"{PREFIX} {KERNEL} at {getattr(aw, '__file__', None)} carries no {fn} (opt_core >= 0.5.19.5)")
    if STATE["precision"] not in aw.PRECISIONS:
        raise RuntimeError(f"{PREFIX} {KERNEL} knows no precision word {STATE['precision']!r}")
    _AW["mod"] = aw
    _patch_block(importlib.import_module(M_DIT))
    _patch_dm_init(importlib.import_module(M_DM))
    STATE["state"] = "on"; STATE["impl"] = getattr(aw, "__file__", None)
    RM.register_user("atom_window", serving, _refuse_lever)
    RM.arm_boundary()
    STATE["installed"] = True
    _log(f"installed: DiffusionTransformerBlock.forward serves the sampler's atom-attention blocks (use_cross_attention) from {KERNEL} "
         f"(ln_qkvg -> window_attn: shifted key windows in place, k/v once per atom, bias+mask+softmax+pv+gates+W_o+residual fused; then the "
         f"conditioned transition) at precision={STATE['precision']} inv={STATE['inv']} from {STATE['impl']}")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
