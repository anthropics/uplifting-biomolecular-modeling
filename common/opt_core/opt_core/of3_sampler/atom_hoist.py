"""The `atom_hoist` lever: the atom path's step-invariant work computed once per rollout instead of once per denoiser step (exact class).

Every denoiser step of the sampler runs the atom-attention encoder and decoder (`DiffusionModule.atom_attn_enc` / `.atom_attn_dec`,
layers/sequence_local_atom_attention.py). Of their work only the noisy-coordinate term `linear_r(r_noisy)` and the attention over the atom
queries depend on the step; the rest is a function of the rollout's inputs (input features, trunk single/pair representations):
`AtomAttentionEncoder.get_atom_reps` — the reference-conformer embedding, `c_l`, the pair activations `p_lm` (incl. `LN + linear` of the
TRUNK pair representation), the pair MLP; each atom transformer's `layer_norm_z(p_lm)` and each of its blocks' pair-bias projection
`linear_z`; the block-index utilities `get_block_indices` / `get_pair_atom_block_mask` (pure functions of the atom mask and window sizes).
This lever memoises exactly those per rollout (`opt_core.of3_sampler.rollout_memo`: shape-keyed, address-stable entries refreshed in place, CUDA-graph
cooperation = one eager denoiser call per rollout) and leaves every stock expression as it is: a memo hit returns the tensor the stock call
produced earlier in the same rollout, so the outputs are bitwise those of the line without the lever (the `exact` line's tier; measured
byte-identical under `--det 1`). `q_l = c_l + linear_r(r_l)` is evaluated per step as stock writes it.

Sites (class / module / instance attributes; no process-wide module hook): `AtomAttentionEncoder.get_atom_reps` (class),
`atom_attention_block_utils.get_block_indices` / `.get_pair_atom_block_mask` (module attributes; their callers live in that module), the two
atom transformers' `layer_norm_z` and their blocks' `attention_pair_bias.linear_z` (instance `forward` attributes, armed at the rollout
boundary from `SampleDiffusion.diffusion_module`), and the rollout boundary itself (rollout_memo). Calls outside a rollout, under grad, or
on a module without the expected attributes run the stock method untouched (counted `outside_calls`).

Published per-rollout buffers (for cells that fuse the atom transformer's blocks): each cross-attention block's
`attention_pair_bias.__dict__["_of3opt_atom_inv"]` = {"epoch", "n_atom", "bias": the block's pair bias as [NB, H, 32, 128] contiguous
(the memo's own storage; the stock caller receives the [NB, 32, 128, H] view of it), and — <KIT>_ATOM_HOIST_INV=full, the default —
the s-side AdaLN terms computed from c_l per atom with the block's own modules: "q_gate"/"q_shift", "k_gate"/"k_shift" (sigmoid(linear_g(LN_s
c_l)), linear_s(LN_s c_l)), "ada_gate" (sigmoid(linear_ada_out(c_l))), "trans_gate"/"trans_shift", "trans_out_gate"}, all [n_atom, C] in
c_l's dtype, valid while `epoch` equals the sampler's `_of3opt_rollout_epoch`; `inv_reference(T, blk, apb, c_l, p_lm, member)` is the stock
expression of one member with the block's own module methods (the definition a consumer, or an outside check, compares against).

Switches (the kit adapter's names, `configure(ENV=…)`; <KIT> = the kit's switch prefix): <KIT>_ATOM_HOIST=1; <KIT>_ATOM_HOIST_MAX_GB (the memo store's byte cap,
default 1.5 — the sampler-memo cap every rollout_memo user shares; ≈0.25 GB at 800 tokens, ≈0.9 GB at 2,000 tokens with the full
namespace); <KIT>_ATOM_HOIST_INV=full|bias. Evidence `<PREFIX> installed …` and one exit line
`<PREFIX> LEVER name=atom_hoist state=on <memo census> sites=get_atom_reps:1,block_utils:2,atom_memos:<n> atom_blocks=<n> inv=<full|bias>`
(atom_blocks = the windowed atom-transformer blocks whose LN_z / linear_z are memoised and whose namespace is published: 6 on both engines; 0 is
named in `errors=no_atom_blocks:…`). Under a live CUDA graph a failure while filling the namespace raises by name (the graph would replay
buffers this rollout did not refresh); on the eager route it is counted (`errors=inv:<Type>`) and the blocks compute their own.
Engines: the OF3 code family (the 0.4.x release and its 0.5.x fork share the class and attribute names read here); the kits'
`cells/atom_hoist.py` are the adapters (switch names, log prefix, the engine's module paths M_SLAA / M_BLOCK_UTILS, registry row).
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict, Optional

from . import rollout_memo as RM

PREFIX = "[opt_core/of3_sampler.atom_hoist]"             # the kit adapter names itself and its switches: configure(PREFIX=, ENV=, ENV_MAX_GB=, ENV_INV=)
ENV: Optional[str] = None                                # <KIT>_ATOM_HOIST=1
ENV_MAX_GB: Optional[str] = None                         # <KIT>_ATOM_HOIST_MAX_GB
ENV_INV: Optional[str] = None                            # <KIT>_ATOM_HOIST_INV=full|bias
VALUES = ("1",)
INV_VALUES = ("full", "bias")
MAX_GB_DEFAULT = 1.5
INV_ATTR = "_of3opt_atom_inv"
M_SLAA: Optional[str] = None                             # the engine module holding AtomAttentionEncoder (….core.model.layers.sequence_local_atom_attention)
M_BLOCK_UTILS: Optional[str] = None                      # the engine module holding get_block_indices / get_pair_atom_block_mask (….core.utils.atom_attention_block_utils)
CONFIGURABLE = ("PREFIX", "ENV", "ENV_MAX_GB", "ENV_INV", "M_SLAA", "M_BLOCK_UTILS")
def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words (log prefix, switch names, the engine's module paths) before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "outside_calls": 0, "errors": {}, "patched": {}, "inv_mode": "full", "atom_blocks": None, "atom_transformers_seen": "",
                         "inv": None, "first": None}
_INV_DICTS = []


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if ("atom_hoist", once_key) in RM._ONCE:
            return
        RM._ONCE.add(("atom_hoist", once_key))
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
    inv = (environ.get(ENV_INV) or "full").strip().lower()
    if inv not in INV_VALUES:
        raise ValueError(f"{ENV_INV}={inv!r} is not one of {'|'.join(INV_VALUES)}")
    mg = (environ.get(ENV_MAX_GB) or "").strip()
    if mg:
        try:
            if float(mg) < 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"{ENV_MAX_GB}={mg!r} is not a non-negative number of GB") from None
    return True


def serving() -> bool:
    return STATE["state"] == "on"


def _refuse(reason: str) -> None:
    if STATE["state"] == "on":
        STATE["state"] = "refused"; STATE["reason"] = reason
        _log(f"REFUSED ({reason}): every atom-path statement runs per step as stock writes it")


def _active() -> bool:
    return serving() and RM.active()


# ----------------------------------------------------------------------------------------------------------------- the published namespace
def _inv(apb) -> dict:
    d = apb.__dict__.get(INV_ATTR)
    if d is None:
        d = {"epoch": -1}
        apb.__dict__[INV_ATTR] = d
        _INV_DICTS.append(d)
    return d


def _clear_inv() -> None:
    for d in _INV_DICTS:
        d.clear(); d["epoch"] = -1


def _memo_pair_bias(apb) -> bool:
    """apb.linear_z memoised per rollout with the permuted, contiguous [.., NB, H, Q, K] layout as STORAGE (what the attention adds); the
    stock caller receives the [.., NB, Q, K, H] view of it, so its own permute lands on the contiguous tensor (a view, no copy)."""
    m = apb.linear_z
    if "forward" in m.__dict__:
        return False
    cls_forward = type(m).forward

    def compute(x):
        out = cls_forward(m, x)                                       # [.., NB, Q, K, H]
        nd = out.dim()
        return (out.permute(*(list(range(nd - 4)) + [nd - 4, nd - 1, nd - 3, nd - 2])).contiguous(),)   # -> [.., NB, H, Q, K]

    def forward(x, *a, **k):
        if a or k or not _active():
            return cls_forward(m, x, *a, **k)
        st = RM.memo_call("pair_bias", ("pair_bias", id(m), RM.sig(x)), lambda: compute(x))[0]
        _inv(apb)["bias"] = st.reshape(-1, *st.shape[-4:])[0] if st.dim() > 4 else st   # [NB, H, Q, K] (lead dims are 1 in the sampler)
        nd = st.dim()
        return st.permute(*(list(range(nd - 4)) + [nd - 4, nd - 2, nd - 1, nd - 3]))  # the stock [.., NB, Q, K, H] view
    forward.__wrapped__ = cls_forward
    m.__dict__["forward"] = forward
    return True


def _fill_inv(enc, cl, plm) -> None:
    """Complete every atom block's namespace for this epoch from c_l [.., A, C] and p_lm — called from get_atom_reps, i.e. before any atom
    transformer block runs (the modules' own later calls hit the same memos). Stock submodule calls, in the caller's context."""
    targets = enc.__dict__.get("_of3opt_atom_inv_targets") or []
    if not targets:
        return
    A = int(cl.shape[-2])
    ep = RM.epoch()
    for T, blk, apb in targets:
        d = _inv(apb)
        if d["epoch"] == ep and d.get("n_atom") == A:
            continue
        if "_of3opt_zln" not in T.__dict__:
            T.__dict__["_of3opt_zln"] = T.layer_norm_z(plm)          # LN_z(p_lm) once per transformer (a memoised instance forward) ...
        apb.linear_z(T.__dict__["_of3opt_zln"])                       # ... and the block's bias memo (sets d["bias"])
        if STATE["inv_mode"] == "full":
            mods = {}
            if hasattr(apb, "layer_norm_a_q") and hasattr(apb.layer_norm_a_q, "linear_g"):
                mods["q"] = (apb.layer_norm_a_q, "adaln")
            if hasattr(apb, "layer_norm_a_k") and hasattr(apb.layer_norm_a_k, "linear_g"):
                mods["k"] = (apb.layer_norm_a_k, "adaln")
            if hasattr(apb, "linear_ada_out"):
                mods["ada"] = ((apb.linear_ada_out, apb.sigmoid), "gate")
            tr = getattr(blk, "conditioned_transition", None)
            if tr is not None and hasattr(tr, "layer_norm") and hasattr(tr.layer_norm, "linear_g"):
                mods["trans"] = (tr.layer_norm, "adaln")
                mods["trans_out"] = ((tr.linear_g, tr.sigmoid), "gate")
            outs = {}
            for name, (m, kind) in mods.items():
                def compute(m=m, kind=kind):
                    if kind == "adaln":
                        sn = m.layer_norm_s(cl)
                        return (m.sigmoid(m.linear_g(sn)).reshape(A, -1), m.linear_s(sn).reshape(A, -1))
                    lin, sg = m
                    return (sg(lin(cl)).reshape(A, -1),)
                outs[name] = RM.memo_call("inv:" + name, ("inv:" + name, id(blk), RM.sig(cl)), compute)   # keyed on the BLOCK + member name
            if "q" in outs:
                d["q_gate"], d["q_shift"] = outs["q"]
            if "k" in outs:
                d["k_gate"], d["k_shift"] = outs["k"]
            if "ada" in outs:
                d["ada_gate"] = outs["ada"][0]
            if "trans" in outs:
                d["trans_gate"], d["trans_shift"] = outs["trans"]
            if "trans_out" in outs:
                d["trans_out_gate"] = outs["trans_out"][0]
        d["epoch"] = ep; d["n_atom"] = A
    for T, _, _ in targets:
        T.__dict__.pop("_of3opt_zln", None)
    if STATE["inv"] is None:
        d0 = _inv(targets[0][2])
        STATE["inv"] = {"blocks": len(targets), "keys": sorted(k for k in d0 if k not in ("epoch", "n_atom")),
                        "shapes": {k: list(v.shape) for k, v in d0.items() if hasattr(v, "shape")}, "n_atom": A}
        _log("%s published on %d atom-attention blocks: %s" % (INV_ATTR, len(targets), STATE["inv"]["shapes"]))


def inv_reference(T, blk, apb, cl, plm, member):
    """The stock expression of one published member with THIS block's own modules (class methods, not the memoised instance attributes)."""
    A = int(cl.shape[-2])
    if member == "bias":
        z = type(T.layer_norm_z).forward(T.layer_norm_z, plm)
        out = type(apb.linear_z).forward(apb.linear_z, z)                        # [.., NB, Q, K, H]
        nd = out.dim()
        out = out.permute(*(list(range(nd - 4)) + [nd - 4, nd - 1, nd - 3, nd - 2])).contiguous()
        return out.reshape(-1, *out.shape[-4:])[0] if out.dim() > 4 else out
    if member in ("q_gate", "q_shift", "k_gate", "k_shift", "trans_gate", "trans_shift"):
        m = {"q": apb.layer_norm_a_q, "k": apb.layer_norm_a_k, "trans": blk.conditioned_transition.layer_norm}[member.split("_")[0]]
        sn = m.layer_norm_s(cl)
        return (m.sigmoid(m.linear_g(sn)) if member.endswith("gate") else m.linear_s(sn)).reshape(A, -1)
    if member == "ada_gate":
        return apb.sigmoid(apb.linear_ada_out(cl)).reshape(A, -1)
    if member == "trans_out_gate":
        tr = blk.conditioned_transition
        return tr.sigmoid(tr.linear_g(cl)).reshape(A, -1)
    raise KeyError(member)


# ----------------------------------------------------------------------------------------------------------------- sites
def _patch_get_atom_reps(Enc) -> bool:
    if getattr(Enc.get_atom_reps, "_of3opt_atom_hoist", False):
        return False
    orig = Enc.get_atom_reps

    def get_atom_reps(self, batch, rl=None, si_trunk=None, zij_trunk=None):
        if rl is None or not _active() or not hasattr(self, "noisy_position_embedder"):
            if serving() and rl is not None and not RM.inside():
                STATE["outside_calls"] += 1
            return orig(self, batch, rl, si_trunk, zij_trunk)
        key = ("atom_reps", id(self), RM.sig(batch["atom_mask"]), RM.sig(zij_trunk), RM.sig(si_trunk))
        holder = {}

        def compute():
            ql, cl, plm = orig(self, batch, rl, si_trunk, zij_trunk)
            holder["ql"] = ql
            return (cl, plm)
        cl, plm = RM.memo_call("atom_reps", key, compute)
        try:
            _fill_inv(self, cl, plm)
        except Exception as e:  # noqa: BLE001
            from ..oom import is_oom
            if is_oom(e):
                raise
            if RM.graphs_active():                                   # a captured graph replays the published buffers: a namespace this rollout did not refresh must not be replayed
                raise RuntimeError(f"{PREFIX} {INV_ATTR} refresh failed under a live CUDA graph ({type(e).__name__}: {e}) — refusing to let the graph "
                                   f"replay last rollout's buffers; run with {ENV} unset") from e
            _count(STATE["errors"], "inv:%s" % type(e).__name__); _log("%s fill failed: %r (the blocks compute their own)" % (INV_ATTR, e), once_key="invfail")
        if "ql" in holder:
            return holder["ql"], cl, plm
        if STATE["first"] is None:
            STATE["first"] = "c_l=%s:%s p_lm=%s" % (tuple(cl.shape), str(cl.dtype).replace("torch.", ""), tuple(plm.shape))
        ql = cl + self.noisy_position_embedder.linear_r(rl)           # the stock expression of q_l on the memoised c_l
        return ql, cl, plm
    get_atom_reps._of3opt_atom_hoist = True; get_atom_reps.__wrapped__ = orig
    Enc.get_atom_reps = get_atom_reps
    return True


def _patch_block_utils(mod) -> int:
    """get_block_indices / get_pair_atom_block_mask: pure functions of (atom mask, window sizes), memoised per rollout by operand signature.
    Their callers are this module's own functions (module-global lookups at call time), so the module attribute is the one site."""
    import torch
    n = 0
    for fname in ("get_block_indices", "get_pair_atom_block_mask"):
        fn = getattr(mod, fname, None)
        if fn is None or getattr(fn, "_of3opt_atom_hoist", False):
            continue

        def mk(fn, fname):
            single = fname == "get_pair_atom_block_mask"

            def wrapped(*args, **kwargs):
                if not _active():
                    return fn(*args, **kwargs)
                try:
                    parts = [fname]
                    for v in list(args) + [kwargs[k] for k in sorted(kwargs)]:
                        parts.append(RM.sig(v) if isinstance(v, torch.Tensor) else (type(v).__name__, str(v)))
                    key = tuple(parts) + (tuple(sorted(kwargs)),)
                except Exception as e:  # noqa: BLE001
                    from ..oom import is_oom
                    if is_oom(e):
                        raise
                    _count(STATE["errors"], "key:%s:%s" % (fname, type(e).__name__))
                    return fn(*args, **kwargs)
                return RM.unwrap(RM.memo_call(fname, key, lambda: fn(*args, **kwargs)), single)
            wrapped._of3opt_atom_hoist = True; wrapped.__wrapped__ = fn; wrapped.__name__ = fname
            return wrapped
        setattr(mod, fname, mk(fn, fname)); n += 1
    return n


def _arm_transformers(sampler) -> None:
    """At the rollout boundary: memoise LN_z and the blocks' linear_z of the two ATOM transformers of the sampler's DiffusionModule
    (cross-attention flavour only; once per instance)."""
    dm = getattr(sampler, "diffusion_module", None)
    if dm is None or getattr(dm, "_of3opt_atom_hoist", False):
        return
    n = 0; targets = []; seen = []
    for name in ("atom_attn_enc", "atom_attn_dec"):
        sub = getattr(dm, name, None)
        T = getattr(sub, "atom_transformer", None) if sub is not None else None
        blocks = list(getattr(T, "blocks", []) or []) if T is not None else []
        # the ATOM transformers are the cross-attention (windowed) ones: a flag on the transformer (0.4.x) or on its blocks (0.5.x); by structure
        # either way — every block's attention module carries a query window (n_query) and its own pair-bias projection (linear_z)
        cross = [b for b in blocks if getattr(getattr(b, "attention_pair_bias", None), "n_query", None) is not None and hasattr(b.attention_pair_bias, "linear_z")]
        seen.append("%s:%d/%d" % (name, len(cross), len(blocks)))
        if T is None or not cross or len(cross) != len(blocks):
            continue
        if hasattr(T, "layer_norm_z"):
            n += RM.memo_instance_forward(T.layer_norm_z, "lnz_plm", serving)
        for blk in cross:
            apb = blk.attention_pair_bias
            n += _memo_pair_bias(apb)
            targets.append((T, blk, apb))
    STATE["atom_blocks"] = len(targets); STATE["atom_transformers_seen"] = ",".join(seen)
    if not targets:                                                  # a sampler without the two windowed atom transformers: named, the prologue memo still serves
        _count(STATE["errors"], "no_atom_blocks:%s" % ",".join(seen))
    enc = getattr(dm, "atom_attn_enc", None)
    if enc is not None:
        enc.__dict__["_of3opt_atom_inv_targets"] = targets           # get_atom_reps (the encoder's) fills every block's namespace from c_l / p_lm
    dm._of3opt_atom_hoist = True
    STATE["patched"]["atom_transformer_memos"] = n
    _log("%d atom-transformer submodules memoised (LN_z(p_lm), per-block linear_z) on %d blocks" % (n, len(targets)))


def census_line() -> str:
    return (f"{PREFIX} LEVER name=atom_hoist state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" {RM.fields()} outside_calls={STATE['outside_calls']} errors={','.join('%s:%d' % kv for kv in sorted(STATE['errors'].items())) or 'none'}"
            f" sites=get_atom_reps:{int(bool(STATE['patched'].get('get_atom_reps')))},block_utils:{STATE['patched'].get('block_utils', 0)},atom_memos:{STATE['patched'].get('atom_transformer_memos', 0)}"
            f" atom_blocks={'pending' if STATE['atom_blocks'] is None else STATE['atom_blocks']}"
            f" inv={STATE['inv_mode']}" + (f" first={STATE['first']}" if STATE["first"] else ""))


def install(environ=None) -> dict:
    """Patch the sites and arm the rollout boundary. Idempotent."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not (M_SLAA and M_BLOCK_UTILS):
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_SLAA=, M_BLOCK_UTILS=) before install")
    import importlib
    slaa = importlib.import_module(M_SLAA); bu = importlib.import_module(M_BLOCK_UTILS)
    STATE["inv_mode"] = (environ.get(ENV_INV) or "full").strip().lower()
    mg = (environ.get(ENV_MAX_GB) or "").strip()
    RM.set_cap_gb(float(mg) if mg else MAX_GB_DEFAULT)
    STATE["patched"]["get_atom_reps"] = _patch_get_atom_reps(slaa.AtomAttentionEncoder)
    STATE["patched"]["block_utils"] = _patch_block_utils(bu)
    STATE["state"] = "on"
    RM.register_user("atom_hoist", serving, _refuse, on_boundary=_arm_transformers, on_drop=_clear_inv)
    RM.arm_boundary()
    STATE["installed"] = True
    _log(f"installed: AtomAttentionEncoder.get_atom_reps (c_l, p_lm), {M_BLOCK_UTILS}.get_block_indices / .get_pair_atom_block_mask and the atom "
         f"transformers' layer_norm_z / per-block linear_z memoised per rollout (rollout_memo: cap {RM.STATE['cap_gb']:.2f} GB, boundaries "
         f"{','.join(RM.STATE['boundaries']) or 'pending'}); {INV_ATTR} mode {STATE['inv_mode']}")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
