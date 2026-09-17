"""confmem -- lever `biasfree`: release the sampler's per-block pair-bias buffers once `sample()` returns.

LEVER `biasfree` -- 6.00 GiB at L=4096.
  Those twelve [1, 16, N, N] bf16 blocks are the DiffusionTransformer's per-block pair-bias buffers,
  cached by `ef2_opt._apb_forward_cached` in `blk._ef2opt_pb` (ef2_opt.py:1371-1434) and registered in
  `ef2_opt._PB_BLOCKS`.  They are read only by the 200 sampling steps.  By the time the confidence head
  runs, `structure_head.sample()` has returned and they are dead for the rest of the fold -- but nothing
  drops them, so 12 x 0.5 GiB sits across the fold's highest peak.  This releases them the instant
  `sample()` returns, through the kit's OWN sanctioned entry point `ef2_opt.clear_graphs()`, which is
  written precisely to "drop graphs + all ptr-referenced static buffers together (never one without the
  other)" (ef2_opt.py:245-249) -- the hazard the module warns about (a captured sampler graph baking in a
  buffer ADDRESS) is exactly what clear_graphs exists to avoid, and EF2_GRAPH_CAPTURE=0 means there are no
  captured graphs to invalidate anyway.  Cost: the next fold allocates its bias buffers again (it already
  RECOMPUTES them every fold -- `_PB_EPOCH["n"]` bumps per fold -- so only the allocation is new).
  Numerics: none.  Same buffers, same values, shorter life.

INSTALL.  `biasfree` wraps `model.structure_head.sample` (an instance method) and needs ef2_opt importable.
"""
import sys
import torch

__version__ = "1.0"
STATS = {"sample_calls": 0, "bias_frees": 0, "bias_freed_GiB": 0.0}
_ORIG = {}


# ------------------------------------------------------------------ biasfree
def _install_biasfree(model):
    sh = getattr(model, "structure_head", None)
    if sh is None:
        raise RuntimeError("confmem: model has no structure_head")
    if "sample" in _ORIG:
        return False
    orig = sh.sample
    _ORIG["sample"] = orig

    dm = getattr(sh, "diffusion_module", None)

    def _count():
        """Bytes held in EVERY place a per-block pair bias can live on this tree."""
        gib = 0.0
        n = 0
        try:
            import ef2_opt as EO
            for blk in list(EO._PB_BLOCKS):                       # kit cache (only when ef2_opt.install(pair_bias_cache=True))
                for ent in list(getattr(blk, "_ef2opt_pb", {}).values()):
                    for t in (ent if isinstance(ent, (tuple, list)) else [ent]):
                        if torch.is_tensor(t):
                            gib += t.untyped_storage().nbytes() / 2**30; n += 1
        except Exception:
            pass
        st = getattr(dm, "_mk_state", None)                        # ef2_mk_sampler's own per-fold bias dict
        for t in list(getattr(st, "pb", {}).values()) if st is not None else []:
            for x in (t if isinstance(t, (tuple, list)) else [t]):
                if torch.is_tensor(x):
                    gib += x.untyped_storage().nbytes() / 2**30; n += 1
        st2 = getattr(dm, "_dit", None)                            # ef2_dit's fused step: the LIVE path under big/opt14_msa.
        if st2 is not None:                                        # biases live in st.hoist[(B,L)]["pb"] / st.cur["pb"] (ef2_dit.py:611, 672),
            _ents = list(st2.hoist.values())                       # NOT in _ef2opt_pb / mk st.pb -- the three places this counter used to look.
            if st2.cur is not None and all(st2.cur is not e for e in _ents):
                _ents.append(st2.cur)
            for _e in _ents:
                for _t in list(_e.get("pb", []) or []) + [_e.get("s0")]:
                    if torch.is_tensor(_t):
                        gib += _t.untyped_storage().nbytes() / 2**30; n += 1
        return gib, n

    def sample(*a, **k):
        out = orig(*a, **k)
        STATS["sample_calls"] += 1
        try:
            import ef2_opt as EO
            gib, n = _count()
            STATS["bias_tensors_seen"] = STATS.get("bias_tensors_seen", 0) + n
            EO.clear_graphs(reason="rc_biasfree")     # graphs + every ptr-referenced static buffer, together
            after, _ = _count()
            STATS["bias_frees"] += 1
            STATS["bias_freed_GiB"] = round(STATS["bias_freed_GiB"] + max(0.0, gib - after), 3)
            STATS["bias_held_GiB"] = round(gib, 3)
        except Exception as e:
            STATS["bias_free_errors"] = int(STATS.get("bias_free_errors", 0)) + 1; STATS["bias_free_last_error"] = repr(e)
        return out
    sh.sample = sample
    return True


def unapply(model=None):
    """Restore ``structure_head.sample`` on ``model`` (when given) and forget the wrapper."""
    orig = _ORIG.pop("sample", None)
    sh = getattr(model, "structure_head", None) if model is not None else None
    if orig is not None and sh is not None:
        sh.sample = orig


def apply(model=None):
    led = {"version": __version__}
    if model is not None:
        led["biasfree"] = _install_biasfree(model)
    return led


def stats():
    return dict(STATS)
