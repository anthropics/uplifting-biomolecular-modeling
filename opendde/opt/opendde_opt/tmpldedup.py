"""tmpl_dedup — the tree's own lever on the stock TemplateEmbedder (exact class, bitwise by construction).

Stock (`opendde/model/modules/pairformer.py:2135-2193`, OpenDDE 1.1.1) `TemplateEmbedder.forward` runs `single_template_forward` — the [N, N, 108]
feature build, `linear_no_bias_z(LN(z)) + linear_no_bias_a(at)`, the c=64 `n_blocks=2` template pair stack (`config/model_base.py:140-143` for
`opendde_v1`) and `layernorm_v` — once per template SLOT, `u = u + v_t` over the slots, then `u / (1e-7 + T)` -> relu -> `linear_no_bias_u`. The
featurizer always pads the slots to the maximum (4): a query without templates (`--use_template false`, the served default) carries 4 bit-identical
dummy slots (`data/inference/infer_dataloader.py:269` `make_dummy_feature`), a query with k real hits carries k distinct + (4-k) identical padding
slots. Identical slots give bit-identical `v_t` (the same kernels on the same operands in one process), so the stock evaluates the same pair stack up
to 4x per recycle x 10 recycles.

The lever: per `forward` call, slots whose five per-slot input features (`SLOT_KEYS`) are bitwise equal (`torch.equal`) form one class; `v` is
computed ONCE per class by the stock `single_template_forward` on the class's first slot, and the stock accumulation loop runs VERBATIM over all T
slots (`u = u + v_class(t)` in slot order, then the stock division / relu / projection) — the addends and their order are the stock's bit for bit, so
`u` is the stock's bit for bit (no algebraic shortcut such as `T * v` is taken). Every other statement is the stock body restated (pinned by
`registry.PIN_STATUS`). The wrapper steps aside BY NAME (delegates to the stock method, counted `bypass:<reason>`) for a call outside the read
surface: `n_blocks < 1` / no `template_aatype` (stock returns 0), a pair tensor that is not [N, N, c], autograd enabled, a per-slot key absent, or slot
tensors whose leading dimension is not the slot count.

Cost of the class test: the slot features are the SAME tensors on every recycle of an item (the trunk hands the one `input_feature_dict` to
each cycle), so the classes are decided ONCE per item — <= 6 pairs x 5 keys of device `eq`+`all` reductions read back in ONE host transfer at
the first recycle (when the host is not ahead of the device) — and memoised for the later recycles on the tensors' identity held by weak
references + their in-place version counters (a freed or mutated tensor can never produce a stale hit: `_memo_lookup`). A host read per recycle
would drain the launch queue the host builds during the 48-block stack and expose the launch-bound template / MSA span that follows it —
hence one read per item. Saving: (T - distinct) template pair-stack passes per recycle
(3 of 4 in a no-template fold). STATS counts calls, slots, distinct, saved passes, class decisions / memo hits, bypasses by reason. Installed from `stack._apply` for the lines that carry the lever through the core's per-site patch
(`opt_core.autoload.patch_attr_at_import`): patched at once when the pairformer module is imported, else right after its import. Idempotent.
"""
from __future__ import annotations

import sys

TARGET = "opendde.model.modules.pairformer"
CLASS, METHOD = "TemplateEmbedder", "forward"
TAG = "opendde-opt"
SLOT_KEYS = ("template_aatype", "template_distogram", "template_pseudo_beta_mask", "template_unit_vector", "template_backbone_frame_mask")
STATS = {"installed": False, "armed": False, "calls": 0, "slots": 0, "distinct": 0, "saved": 0, "decisions": 0, "memo_hits": 0, "bypass": 0, "bypass_reasons": {}}
_MEMO = {"refs": None, "versions": None, "rep": None}                     # the last class decision: weakrefs to the SLOT_KEYS tensors, their _version counters, rep


class ActivationError(RuntimeError):
    """The pairformer module has no `TemplateEmbedder.forward` to wrap: the kit's activation fails by name."""


def _bypass(reason: str) -> None:
    STATS["bypass"] += 1
    STATS["bypass_reasons"][reason] = STATS["bypass_reasons"].get(reason, 0) + 1


def slot_classes(feats: dict, n_slots: int, equal) -> list:
    """rep[t] = the first slot s <= t whose SLOT_KEYS tensors all compare `equal` to slot t's (rep[t] == t for a class's first member)."""
    rep = []
    for t in range(n_slots):
        r = t
        for s in range(t):
            if rep[s] != s:                                   # compare against class representatives only
                continue
            if all(equal(feats[k][t], feats[k][s]) for k in SLOT_KEYS):
                r = s
                break
        rep.append(r)
    return rep


def slot_classes_device(feats: dict, n_slots: int, torch) -> list:
    """`slot_classes` decided on the device with ONE host read: every pair (s < t) x key `eq().all()` flag stacked and transferred once."""
    if n_slots <= 1:
        return list(range(n_slots))
    pairs = [(s_, t) for t in range(n_slots) for s_ in range(t)]
    flags = torch.stack([torch.stack([(feats[k][t] == feats[k][s_]).all() for k in SLOT_KEYS]).all() for (s_, t) in pairs])
    same = dict(zip(pairs, flags.tolist()))                       # the one device -> host transfer of the decision
    return slot_classes({k: list(range(n_slots)) for k in SLOT_KEYS}, n_slots, lambda t_, s__: same[(min(t_, s__), max(t_, s__))])


def _memo_lookup(feats: dict):
    """The memoised rep when the SLOT_KEYS tensors are the very objects of the last decision (alive weak references) at the same in-place
    versions; else None. Never reads the device."""
    refs, vers = _MEMO["refs"], _MEMO["versions"]
    if refs is None:
        return None
    for k, r, v in zip(SLOT_KEYS, refs, vers):
        t = feats.get(k)
        if r() is not t or getattr(t, "_version", None) != v:
            return None
    return _MEMO["rep"]


def _memo_store(feats: dict, rep: list) -> None:
    import weakref
    try:
        _MEMO["refs"] = tuple(weakref.ref(feats[k]) for k in SLOT_KEYS)
        _MEMO["versions"] = tuple(getattr(feats[k], "_version", None) for k in SLOT_KEYS)
        _MEMO["rep"] = list(rep)
    except TypeError:                                             # an object that takes no weak reference: no memo (decide every call)
        _MEMO["refs"] = _MEMO["versions"] = _MEMO["rep"] = None


def classes_for(feats: dict, n_slots: int, torch) -> list:
    """rep for this call: the memo when valid, else one device decision (counted)."""
    rep = _memo_lookup(feats)
    if rep is not None and len(rep) == n_slots:
        STATS["memo_hits"] += 1
        return rep
    rep = slot_classes_device(feats, n_slots, torch)
    STATS["decisions"] += 1
    _memo_store(feats, rep)
    return rep


def make_wrapper(orig):
    """The method installed on the class: the stock forward with each class of identical slots embedded once (bitwise = stock)."""
    import torch

    def forward(self, input_feature_dict, z, pair_mask=None, triangle_attention="torch", triangle_multiplicative="torch",
                inplace_safe=False, chunk_size=None):
        # ---- the stand-aside surface (delegations, counted by name) -------------------------------------------------------------
        if "template_aatype" not in input_feature_dict or getattr(self, "n_blocks", 0) < 1:
            _bypass("no_templates_or_zero_blocks")
            return orig(self, input_feature_dict, z, pair_mask=pair_mask, triangle_attention=triangle_attention,
                        triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
        if torch.is_grad_enabled() or not torch.is_tensor(z) or z.dim() != 3:
            _bypass("grad_enabled" if torch.is_grad_enabled() else "pair_not_NxNxc")
            return orig(self, input_feature_dict, z, pair_mask=pair_mask, triangle_attention=triangle_attention,
                        triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
        num_templates = input_feature_dict["template_aatype"].shape[0]
        for k in SLOT_KEYS:
            t = input_feature_dict.get(k)
            if not torch.is_tensor(t) or t.dim() < 1 or t.shape[0] != num_templates:
                _bypass(f"slot_key_unreadable:{k}")
                return orig(self, input_feature_dict, z, pair_mask=pair_mask, triangle_attention=triangle_attention,
                            triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
        # ---- the stock body (pairformer.py:2164-2193), the slot loop de-duplicated ---------------------------------------------
        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)
        num_residues = z.shape[0]
        query_num_channels = z.shape[-1]
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])
        z = self.layernorm_z(z)
        rep = classes_for(input_feature_dict, num_templates, torch)
        v_of = {}
        for template_id in range(num_templates):
            if rep[template_id] == template_id:                 # a class's first slot: the stock evaluation
                v_of[template_id] = self.single_template_forward(
                    template_id=template_id, input_feature_dict=input_feature_dict, z=z, pair_mask=pair_mask,
                    multichain_mask=multichain_mask, triangle_attention=triangle_attention,
                    triangle_multiplicative=triangle_multiplicative, inplace_safe=inplace_safe, chunk_size=chunk_size)
        u = 0
        for template_id in range(num_templates):                # the stock accumulation, slot order and addends unchanged
            u = u + v_of[rep[template_id]]
        u = u / (1e-7 + num_templates)
        u = self.linear_no_bias_u(self.relu(u))
        assert u.shape == (num_residues, num_residues, query_num_channels)
        distinct = len(v_of)
        STATS["calls"] += 1
        STATS["slots"] += int(num_templates)
        STATS["distinct"] += distinct
        STATS["saved"] += int(num_templates) - distinct
        return u

    forward._tmpl_dedup = True
    forward._orig = orig
    return forward


def _sync() -> None:
    p = STATS.get("patch")
    if p is not None:
        STATS["installed"], STATS["armed"] = p.state == "installed", p.state == "armed"


def install() -> None:
    """Patch now when the pairformer module is imported, else at its import (the core's per-site patch). Idempotent."""
    _sync()
    if STATS["installed"] or STATS["armed"]:
        return
    from opt_core import autoload
    try:
        STATS["patch"] = autoload.patch_attr_at_import(TARGET, f"{CLASS}.{METHOD}", make_wrapper, tag=TAG, name="tmpl_dedup")
    except autoload.PatchError as e:
        raise ActivationError(str(e)) from None
    _sync()


def _reset() -> None:
    """Test support: this module's process state back to import time — an armed patch withdrawn, an installed one restored, the core's
    per-site record dropped (the next install() starts afresh), counters and memo cleared."""
    patch = STATS.get("patch")
    if patch is not None:
        for undo in (getattr(patch, "disarm", None), getattr(patch, "restore", None)):
            try:
                undo and undo()
            except Exception:  # noqa: BLE001
                pass
    try:
        from opt_core import autoload
        autoload._PATCHES.pop((TARGET, f"{CLASS}.{METHOD}"), None)
    except Exception:  # noqa: BLE001
        pass
    STATS["patch"] = None
    for k in ("calls", "slots", "distinct", "saved", "decisions", "memo_hits", "bypass"):
        STATS[k] = 0
    STATS["bypass_reasons"].clear(); STATS["installed"] = STATS["armed"] = False
    _MEMO.update(refs=None, versions=None, rep=None)


def kit_stats() -> dict:
    _sync()
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in STATS.items() if k != "patch"}


def fallbacks(planned) -> list:
    """The lever's named events at exit: the pairformer module imported but the class not wrapped. A bypass is the lever's declared
    stand-aside surface (named in STATS, not a fallback); a process that never imported the module has no event."""
    if "tmpl_dedup" not in (planned or ()):
        return []
    _sync()
    out = []
    m = sys.modules.get(TARGET)
    if m is not None and not STATS["installed"]:
        out.append("tmpl_dedup: the pairformer module is imported but TemplateEmbedder.forward is not wrapped")
    elif m is not None:
        f = getattr(getattr(m, CLASS, None), METHOD, None)
        if f is not None and not getattr(f, "_tmpl_dedup", False):
            out.append("tmpl_dedup: TemplateEmbedder.forward was re-bound after the lever (another unit owns the site)")
    return out
