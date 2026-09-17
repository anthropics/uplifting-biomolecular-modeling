"""ef2_conf — the memory mode's confidence-head word: with ``num_diffusion_samples = S > 1`` the confidence head (its own pair trunk + the
pLDDT / PAE / PDE heads, upstream ``ConfidenceHead.forward``) runs ONCE PER STRUCTURE SAMPLE at S = 1 instead of once on the S-fold
repeated batch, and the per-sample outputs are assembled along dim 0 in upstream's own ``b*S + s`` order (``_repeat_batch`` /
``_flatten_sample_axis``), so every key keeps its stock shape.  The batched statement materialises ``[B*S, N, N, 256]`` fp32 pair tensors
(19.1 GiB at 2000 tokens x 5 samples — the memory mode ran out of memory there on an 80 GB card); the per-sample statement never holds more
than one sample's pair tensors.  Arithmetic per sample is upstream's own S = 1 statement (tolerance class: a batched GEMM and a single GEMM
may pick different library kernels).  S == 1: the upstream statement itself, untouched.

Switch: ``EF2_CONF_PER_SAMPLE`` = "1" (the memory mode exports it: esmfold2_opt.modes.BIG_FAST.conf_per_sample) / "0" (off; a caller's
"0" wins).  ``install(model)`` patches the model's ``confidence_head`` instance; ``uninstall()`` restores it; ``describe()`` / ``stats()``
report.  The per-sample ``pae_logits`` / ``pde_logits`` are not kept (None in the output; the row-sharded route's contract).  The XL add-on's storage frees inside ``ConfidenceHead.forward`` (x2b: z / relpos / token-bond encodings released after their last
consumer) compose: every sample but the last receives its own copy of those three inputs, the last one the caller's tensors, so the add-on
releases the caller's storage exactly once, after the last consumer.
"""
import os
import sys

import torch

ENV = "EF2_CONF_PER_SAMPLE"
CONF_LOGITS_DROPPED = ("pae_logits", "pde_logits")        # the per-sample [1, N, N, 64] fp32 logits are not kept across the loop (the row-sharded route's
                                                          # contract, esmfold2_opt.rowpair_heads.CONF_LOGITS_DROPPED: upstream's processor and the kit's
                                                          # writer read neither; `pae` / `pde` themselves are kept): those keys are None in the output
STATS = {"calls": 0, "per_sample_calls": 0, "samples": 0, "batched_calls": 0, "clones": 0, "atom_statics_released_gib": 0.0, "logits_dropped": 0}
_STATE = {"installed": False, "head": None, "orig": None}


def _mod():
    import transformers.models.esmfold2.modeling_esmfold2 as M
    return M


def _frees():
    """The XL add-on's active free set inside ConfidenceHead.forward ('z', 'relpos'), or () when the add-on is not installed."""
    cfg = getattr(_mod(), "_EF2XL_CFG", None)
    try:
        return set(cfg.get("free", ()) if cfg else ())
    except Exception:
        return set()


def _assemble(outs, B, S):
    """Per-sample output dicts -> one dict in upstream's b*S + s order along dim 0 (keys without a leading batch axis: the first sample's)."""
    res = {}
    for k, v0 in outs[0].items():
        if torch.is_tensor(v0) and v0.dim() >= 1 and int(v0.shape[0]) == B:
            res[k] = torch.stack([o[k] for o in outs], dim=1).reshape(B * S, *v0.shape[1:])
        else:
            res[k] = v0
    return res


def _forward_per_sample(self, s_inputs, z, x_pred, distogram_atom_idx, token_attention_mask, atom_to_token, atom_attention_mask, asym_id, mol_type,
                        num_diffusion_samples=1, relative_position_encoding=None, token_bonds_encoding=None):
    cls_forward = type(self).forward                      # the class statement in force (upstream's, or the XL add-on's patched source)
    S = int(num_diffusion_samples)
    STATS["calls"] += 1
    atom = sys.modules.get("ef2_atom")                    # the sampler returned: its atom-path static buffers are dead in a graph-free process
    if atom is not None and hasattr(atom, "release_statics_if_eager"):
        STATS["atom_statics_released_gib"] = round(STATS.get("atom_statics_released_gib", 0.0) + float(atom.release_statics_if_eager("confidence")), 3)
    kw = dict(distogram_atom_idx=distogram_atom_idx, token_attention_mask=token_attention_mask, atom_to_token=atom_to_token,
              atom_attention_mask=atom_attention_mask, asym_id=asym_id, mol_type=mol_type)
    B = int(z.shape[0])
    x4 = None                                             # x_pred with an explicit sample axis [B, S, atoms, 3]: upstream accepts [B, S, atoms, 3] or the
    if S > 1:                                             # already-flat [B*S, atoms, 3] (its _flatten_sample_axis reshapes only the 4-D form; b*S + s order)
        if x_pred.dim() == 4 and int(x_pred.shape[0]) == B and int(x_pred.shape[1]) == S:
            x4 = x_pred
        elif x_pred.dim() == 3 and int(x_pred.shape[0]) == B * S:
            x4 = x_pred.reshape(B, S, *x_pred.shape[1:])
    if x4 is None:
        STATS["batched_calls"] += 1                      # S == 1 (or a layout this word does not restate): the statement as is
        return cls_forward(self, s_inputs, z, x_pred, num_diffusion_samples=num_diffusion_samples,
                           relative_position_encoding=relative_position_encoding, token_bonds_encoding=token_bonds_encoding, **kw)
    frees = _frees()
    outs = []
    for s in range(S):
        last = s == S - 1
        zz, rp, tb = z, relative_position_encoding, token_bonds_encoding
        if not last:                                      # the add-on frees these inside the call: hand every sample but the last its own copy
            if "z" in frees:
                zz = z.clone(); STATS["clones"] += 1
            if "relpos" in frees:
                if rp is not None:
                    rp = rp.clone(); STATS["clones"] += 1
                if tb is not None:
                    tb = tb.clone(); STATS["clones"] += 1
        out_s = cls_forward(self, s_inputs, zz, x4[:, s:s + 1], num_diffusion_samples=1,
                            relative_position_encoding=rp, token_bonds_encoding=tb, **kw)
        for k in CONF_LOGITS_DROPPED:                     # (S-1) x N^2 x 64 fp32 never resident across samples (2.9 GiB per sample at 3000 tokens)
            if isinstance(out_s, dict) and out_s.get(k) is not None:
                out_s[k] = None; STATS["logits_dropped"] += 1
        outs.append(out_s)
        del zz, rp, tb, out_s
        STATS["per_sample_calls"] += 1
    STATS["samples"] += S
    return _assemble(outs, B, S)


def enabled_by_env():
    return os.environ.get(ENV, "0") == "1"


def install(model):
    """Patch ``model.confidence_head.forward`` (instance) with the per-sample statement; returns a census string."""
    head = getattr(model, "confidence_head", None)
    if head is None:
        return "no confidence_head on the model: nothing patched"
    if _STATE["installed"] and _STATE["head"] is head:
        return "already installed"
    import types
    _STATE.update(installed=True, head=head, orig=head.__dict__.get("forward"))
    head.forward = types.MethodType(_forward_per_sample, head)
    print(f"[ef2_conf] confidence head per structure sample: on ({ENV}=1; num_diffusion_samples > 1 runs upstream's S=1 statement once per sample, "
          f"outputs assembled in upstream's b*S+s order; XL frees composed: {sorted(_frees()) or 'none'})", flush=True)
    return "installed (instance patch on model.confidence_head)"


def uninstall():
    head = _STATE.get("head")
    if head is not None and _STATE["installed"]:
        if _STATE["orig"] is None:
            head.__dict__.pop("forward", None)
        else:
            head.forward = _STATE["orig"]
    _STATE.update(installed=False, head=None, orig=None)


def describe():
    return {"installed": bool(_STATE["installed"]), "env": os.environ.get(ENV), **STATS}


def stats():
    return dict(STATS)
