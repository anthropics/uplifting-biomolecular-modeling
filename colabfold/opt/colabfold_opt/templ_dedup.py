"""TEMPL_DEDUP — identical templates embedded once (strategy LOCAL.colabfold.template_dedup).

What stock does. AlphaFold-Multimer's ``TemplateEmbedding`` (``alphafold/model/modules_multimer.py:775-852``) runs ``SingleTemplateEmbedding``
— the template pair features, 2 template-pair-stack blocks at c_t = 64 over the N x N plane, the output LayerNorm — once PER TEMPLATE ROW
inside an ``hk.scan`` (``:835-843``: ``carry + E(template_i)``), every recycle, then ``/ num_templates``, ReLU, ``output_linear``
(``:845-850``). colabfold's default run (no ``--templates``) gives the model template rows that are all IDENTICAL: ``colabfold/batch.py:96-130``
``mk_mock_template`` — a poly-'A' sequence (one-hot 'A' = row 0), all-zero atom positions and masks — and AlphaFold's multimer feature
pipeline pads the template axis to ``MAX_TEMPLATES = 4`` with zeros (``alphafold/data/feature_processing.py:35, 71-77``), a zero row
being that same mock row again (aatype 0, positions 0, mask 0): four equal rows, four equal embeddings, three of them recomputed for nothing.

What the lever does, without editing stock: ``enable()`` rebinds ``modules_multimer.TemplateEmbedding`` (looked up as a module global by
``EmbeddingsAndEvoformer`` at trace time, ``:642``) to the subclass below. Its ``__call__`` is stock's body with ONE change inside the
scan: row i is compared ON THE DEVICE with row i-1 (aatype, atom positions, atom mask — exact equality) and, when equal, row i-1's
embedding is REUSED through ``hk.cond`` instead of recomputed (XLA runs only the taken branch). The sum keeps stock's order and operands
(``carry + E`` per row, then ``/ num_templates``): E of an equal row IS the value the previous row produced, so the reuse itself changes no
value; the ``hk.cond`` compute branch is compiled apart from stock's plain scan body, so a computed row may differ from stock's at the
rounding level — Tier 2 with `fast`. One program serves every input: rows that differ (real templates found with
``--templates``, ``:702-725``) take the compute branch each — that query costs what stock costs (+ T-1 row comparisons) and the lever has
stepped aside for it BY THE DATA, which its census line says; nothing is refused. The row comparison is the whole test: a template row equal
to its predecessor in all three arrays has, by ``SingleTemplateEmbedding``'s definition (a function of the row, the query embedding and the
masks, at inference no dropout), the same embedding.

The census is read on the host, once per predict, by a counting subclass of ``alphafold.model.model.RunModel`` (``predict`` sees the numpy
feature dict before upload; no device work, no synchronisation added to the graph): per predict the template rows T and the distinct
consecutive rows U the scan will embed. A predict the lever cannot serve is counted under its reason and runs stock's body untouched:
``monomer_route`` (a single-chain query under ``--model-type auto`` runs alphafold2_ptm, ``alphafold.model.modules.TemplateEmbedding`` —
attention over templates, another body; not rebound), ``template_disabled`` (a model configuration without the template embedder) and
``dropout_enabled`` (stock's ``--use-dropout`` = ``global_config.eval_dropout``: equal rows are then independent dropout draws, not one value —
the subclass runs stock's body, uncounted) — all in ``registry.STEP_ASIDE_RULES``: by design, never a partial activation. Under `big` at ``--n_gpu`` P > 1 the row-sharded template
region owns ``TemplateEmbedding.__call__`` (opt_core.mem.rowpair_jax.alphafold_template) and this one-device lever steps aside by name
(``state=off reason=n_gpu>1``), as TRIMUL_PALLAS does.

Evidence: ``_STATE`` — ``calls`` (multimer ``TemplateEmbedding`` traces served by the dedup body), ``predicts`` (predict calls seen),
``templates`` (template rows over those predicts), ``embedded`` (rows the scan embeds = distinct consecutive rows), ``reused`` (rows that
reuse their predecessor's embedding), ``unique_hist`` (``<U>:<predicts>,…`` — ``1:3`` on a default three-predict run, ``4:1`` on a query
with four real templates), ``fallbacks`` / ``fallback_by`` (``monomer_route:<n>``, ``template_disabled:<n>``). Markers
``modules_multimer.TemplateEmbedding._templ_dedup`` and ``model.RunModel._templ_dedup_census``.
"""
from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Optional

NAME = "TEMPL_DEDUP"
STRATEGY = "LOCAL.colabfold.template_dedup"
MODULES_MULTIMER = "alphafold.model.modules_multimer"
MODEL_MODULE = "alphafold.model.model"
CLASS = "TemplateEmbedding"
MODEL_CLASS = "RunModel"
MARKER = "_templ_dedup"
CENSUS_MARKER = "_templ_dedup_census"
MONOMER_ROUTE, TEMPLATE_DISABLED, DROPOUT_ENABLED = "monomer_route", "template_disabled", "dropout_enabled"      # (+ dropout_enabled: under stock's --use-dropout = global_config.eval_dropout equal rows are independent dropout draws, not one value — stock's body runs) registry.STEP_ASIDE_RULES: the predicts the lever cannot serve, by name
N_GPU_REASON = "n_gpu>1"                    # big at P > 1: the row-sharded template region owns TemplateEmbedding.__call__; this one-device lever is dropped there
FEATURES = ("template_aatype", "template_all_atom_positions", "template_all_atom_mask")   # the three per-row arrays the scan maps (modules_multimer.py:841-843), the multimer feature names

_STATE: Dict[str, Any] = {"enabled": False, "calls": 0, "predicts": 0, "templates": 0, "embedded": 0, "reused": 0, "unique_hist": "none",
                          "fallbacks": 0, "fallback_by": "none", "sites": "multimer"}
_COUNTS: Dict[str, Any] = {"fallback_by": {}, "unique_hist": {}}
_ORIG: Dict[str, Any] = {"cls": None, "mm": None, "model_cls": None, "model": None}


def _render(d) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))) or "none"


def consecutive_unique(template_batch) -> Optional[tuple]:
    """``(T, U)`` for a feature dict: T template rows, U = the rows the dedup scan embeds (row 0 and every row that differs from its
    predecessor in any of the three arrays — exact equality, as the device compares them); None when the dict lacks the three arrays."""
    import numpy as np  # noqa: PLC0415
    try:
        arrs = [np.asarray(template_batch[k]) for k in FEATURES]
    except (KeyError, TypeError):
        return None
    t = int(arrs[0].shape[0]) if arrs[0].ndim >= 1 else 0
    if any(int(a.shape[0]) != t for a in arrs if a.ndim >= 1):
        return None
    u = 0
    for i in range(t):
        if i == 0 or any(not np.array_equal(a[i], a[i - 1]) for a in arrs):
            u += 1
    return t, u


def census(feat, config=None) -> Optional[str]:
    """Count one predict on the host: the step-aside reason (returned) when the lever cannot serve this predict, else None with T / U tallied."""
    _STATE["predicts"] += 1
    reason = None
    try:
        gc = config.model.global_config if config is not None else None
        if gc is not None and not bool(gc.get("multimer_mode", False)):
            reason = MONOMER_ROUTE
        elif config is not None and not bool(config.model.embeddings_and_evoformer.template.get("enabled", True)):
            reason = TEMPLATE_DISABLED
        elif gc is not None and bool(gc.get("eval_dropout", False)):
            reason = DROPOUT_ENABLED
    except (AttributeError, KeyError, TypeError):
        reason = None
    if reason is None:
        tu = consecutive_unique(feat) if isinstance(feat, dict) else None
        if tu is None:
            reason = MONOMER_ROUTE if isinstance(feat, dict) and "template_all_atom_masks" in feat else None   # the monomer feature dict names the mask in the plural
        else:
            t, u = tu
            _STATE["templates"] += t; _STATE["embedded"] += u; _STATE["reused"] += t - u
            h = _COUNTS["unique_hist"]; h[u] = h.get(u, 0) + 1; _STATE["unique_hist"] = _render(h)
            return None
    if reason is not None:
        _STATE["fallbacks"] += 1
        fb = _COUNTS["fallback_by"]; fb[reason] = fb.get(reason, 0) + 1; _STATE["fallback_by"] = _render(fb)
    return reason


def _dropout(global_config) -> bool:
    """True when the model applies dropout at inference (colabfold `--use-dropout` sets ``global_config.eval_dropout``)."""
    try:
        return bool(global_config.get("eval_dropout", False)) if hasattr(global_config, "get") else bool(getattr(global_config, "eval_dropout", False))
    except (AttributeError, TypeError):
        return False


def _build(stock_cls):
    """The dedup subclass of ``modules_multimer.TemplateEmbedding``: stock's ``__call__`` (modules_multimer.py:808-852) with the scan body changed."""
    import haiku as hk  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    mm = importlib.import_module(MODULES_MULTIMER)
    prng, common_modules = mm.prng, mm.common_modules

    class TemplateEmbedding(stock_cls):                         # defined in a class body so haiku's metaclass wraps __call__ (the module's own name / parameter scope)
        def __call__(self, query_embedding, template_batch, padding_mask_2d, multichain_mask_2d, is_training, safe_key=None):
            if is_training or _dropout(self.global_config):           # dropout live (stock's --use-dropout = global_config.eval_dropout, or training): equal rows are
                return stock_cls.__call__(self, query_embedding, template_batch, padding_mask_2d, multichain_mask_2d, is_training, safe_key)   # independent draws — stock's body, uncounted (census: dropout_enabled)
            # ---- stock :808-833, verbatim
            c = self.config
            if safe_key is None:
                safe_key = prng.SafeKey(hk.next_rng_key())
            num_templates = template_batch['template_aatype'].shape[0]
            num_res, _, query_num_channels = query_embedding.shape
            template_embedder = mm.SingleTemplateEmbedding(self.config, self.global_config)

            def partial_template_embedder(template_aatype, template_all_atom_positions, template_all_atom_mask, unsafe_key):
                safe_key = prng.SafeKey(unsafe_key)
                return template_embedder(query_embedding, template_aatype, template_all_atom_positions, template_all_atom_mask,
                                         padding_mask_2d, multichain_mask_2d, is_training, safe_key)

            safe_key, unsafe_key = safe_key.split()
            unsafe_keys = jax.random.split(unsafe_key._key, num_templates)
            _STATE["calls"] += 1
            # ---- the change: the carry holds (sum, previous row's embedding, previous row); a row equal to its predecessor reuses that embedding
            aat, pos, msk = (template_batch[k] for k in FEATURES)

            def scan_fn(carry, x):
                summed, prev_emb, prev_aat, prev_pos, prev_msk = carry
                t_aat, t_pos, t_msk, t_key = x
                same = jnp.all(t_aat == prev_aat) & jnp.all(t_pos == prev_pos) & jnp.all(t_msk == prev_msk)
                emb = hk.cond(same,
                              lambda o: o[0],                                                                   # reuse: nothing runs
                              lambda o: partial_template_embedder(o[1], o[2], o[3], o[4]).astype(o[0].dtype),   # compute (stock's call; the cast is the identity when the dtypes agree, as stock's `carry + E` requires)
                              (prev_emb, t_aat, t_pos, t_msk, t_key))
                return (summed + emb, emb, t_aat, t_pos, t_msk), None

            zeros = jnp.zeros((num_res, num_res, c.num_channels), dtype=query_embedding.dtype)      # stock's scan_init (:838-839), twice: the sum and the "previous embedding" slot
            scan_init = (zeros, zeros, jnp.full(aat.shape[1:], -1, aat.dtype),                    # previous row = a sentinel no template row equals (aatype -1): row 0 always computes
                         jnp.zeros(pos.shape[1:], pos.dtype), jnp.zeros(msk.shape[1:], msk.dtype))
            (summed_template_embeddings, _e, _a, _p, _m), _ = hk.scan(scan_fn, scan_init, (aat, pos, msk, unsafe_keys))
            # ---- stock :845-852, verbatim
            embedding = summed_template_embeddings / num_templates
            embedding = jax.nn.relu(embedding)
            embedding = common_modules.Linear(query_num_channels, initializer='relu', name='output_linear')(embedding)
            return embedding

    setattr(TemplateEmbedding, MARKER, True)
    TemplateEmbedding._templ_dedup_wrapped = stock_cls
    TemplateEmbedding.__qualname__ = TemplateEmbedding.__name__ = CLASS
    TemplateEmbedding.__module__ = stock_cls.__module__
    return TemplateEmbedding


def _build_census(base):
    """The counting subclass of ``model.RunModel``: ``predict`` counts the feature dict on the host (census) and runs the class it wraps."""

    class RunModel(base):                        # noqa: D401 — the stock name, so repr / logging read the same
        def predict(self, feat, *args, **kwargs):
            census(feat, getattr(self, "config", None))
            return super().predict(feat, *args, **kwargs)

    setattr(RunModel, CENSUS_MARKER, True)
    RunModel.__qualname__ = RunModel.__name__ = MODEL_CLASS
    RunModel.__module__ = base.__module__
    return RunModel


def enable(multimer=None, model=None) -> Dict[str, Any]:
    """Rebind ``modules_multimer.TemplateEmbedding`` to the dedup subclass and ``model.RunModel`` to the counting subclass (idempotent; whatever
    class is bound there now is wrapped — DEVICE_RESIDENT's when it was applied first, the mode table's order). ``multimer`` / ``model``: the
    target modules (tests pass stubs); None imports them."""
    mm = multimer if multimer is not None else importlib.import_module(MODULES_MULTIMER)
    cur = getattr(mm, CLASS)
    if not getattr(cur, MARKER, False):
        setattr(mm, CLASS, _build(cur)); _ORIG.update(cls=cur, mm=mm)
    md = model if model is not None else importlib.import_module(MODEL_MODULE)
    base = getattr(md, MODEL_CLASS)
    if not getattr(base, CENSUS_MARKER, False):
        setattr(md, MODEL_CLASS, _build_census(base)); _ORIG.update(model_cls=base, model=md)
    _STATE["enabled"] = True
    return dict(_STATE)


def disable() -> None:
    if _ORIG["mm"] is not None and _ORIG["cls"] is not None:
        setattr(_ORIG["mm"], CLASS, _ORIG["cls"])
    if _ORIG["model"] is not None and _ORIG["model_cls"] is not None:
        setattr(_ORIG["model"], MODEL_CLASS, _ORIG["model_cls"])
    _STATE["enabled"] = False


def marker_present(multimer=None) -> Optional[bool]:
    m = multimer if multimer is not None else sys.modules.get(MODULES_MULTIMER)
    if m is None:
        return None
    return bool(getattr(getattr(m, CLASS, None), MARKER, False))


def reset_for_tests() -> None:
    disable()
    _STATE.update(enabled=False, calls=0, predicts=0, templates=0, embedded=0, reused=0, unique_hist="none", fallbacks=0, fallback_by="none")
    _COUNTS.update(fallback_by={}, unique_hist={}); _ORIG.update(cls=None, mm=None, model_cls=None, model=None)
