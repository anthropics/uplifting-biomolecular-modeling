"""big_levers.py — the memory levers of this engine's ``big`` mode: registered in ``opt_core.mem.registry`` (framework ``jax``), applied in the
MODEL process by big_launch.py. Module level imports only the standard library and ``opt_core.mem``; ``jax``/``haiku``/``alphafold3`` load inside
each lever's ``applies``/``apply``. Each lever checks its hook points by name (missing or changed = refusal, never a silent no-op) and patches only
the site it owns, reusing a library hook where one applies and transcribing its own JAX-native logic — explicit keys, no torch RNG — where the
generic cannot. Levers, all applied in --mode big: samples_per_pass, transition_shard, trimul_chunk, logits_shard, cond_shard (under --n_gpu P > 1
the wrapper switches off the three the row-sharded pair stack supersedes, modes.with_n_gpu)."""
from __future__ import annotations

import functools

import os
from typing import Optional

from opt_core.mem import registry
from opt_core.oom import is_oom
from opt_core.mem import ckpt as _core_ckpt  # noqa: F401  the core's lever modules register first: its torch `samples_per_pass` (opt_core.mem.ckpt) is replaced by name below by this engine's JAX transcription of the same lever (one name, one flag, one record field, shared with the core)
from opt_core.mem.registry import Applied, refuse

FRAMEWORK = "jax"
SAMPLES_PER_PASS, TRANSITION_SHARD, COND_SHARD = "samples_per_pass", "transition_shard", "cond_shard"
LOGITS_SHARD = "logits_shard"
TRIMUL_CHUNK = "trimul_chunk"
LINE = (SAMPLES_PER_PASS, TRANSITION_SHARD, COND_SHARD, LOGITS_SHARD, TRIMUL_CHUNK)   # the order of application (every lever of the kit; big.LEVERS spells the same tuple)
LINE_LEVERS = {"big": LINE}                                               # the levers per line, in order: the mode's composition = every lever; modes.BIG_LINES names the base per line
HOOKS = {                                                                   # the engine's hook points, by name (the fork bc32b22f; checked at applies()) --
    # qualnames only, no digests of in-repo content (the version is the git commit; stock/src is never edited in place, so the transcription
    # base can only change through a deliberate, reviewed pin-bump commit — never silently). Staleness is instead caught by behavioural
    # equivalence / signature-existence tests (test_big.py), not a runtime recompute-and-compare gate.
    SAMPLES_PER_PASS: {"module": "alphafold3.model.network.diffusion_head", "attr": "sample"},
    TRANSITION_SHARD: {"module": "alphafold3.model.model", "cls": "Model", "path": "global_config.pair_transition_shard_spec"},
    COND_SHARD: {"module": "alphafold3.model.network.diffusion_head", "cls": "DiffusionHead", "method": "_conditioning",
                 "mapping": "alphafold3.model.components.mapping",
                 "hoist_module": "af3_flashpairformer.diffusion_hoist", "hoist_attr": "_pair_conditioning",              # the FlashPairformer hoist's precompute path (patched too when its module is loaded)
                 "rel_module": "alphafold3.model.network.featurization", "rel_attr": "create_relative_encoding"},
    TRIMUL_CHUNK: {"module": "alphafold3.model.network.modules", "cls": "TriangleMultiplication", "method": "__call__",
                   "glut_module": "af3_pallas_levers", "glut_attr": "glu_transposed_masked", "glut_factory": "make_patched_trimul_class"},
    LOGITS_SHARD: {"module": "alphafold3.model.network.diffusion_transformer", "cls": "Transformer", "method": "__call__",
                   "mapping": "alphafold3.model.components.mapping"},
}
SETTINGS = {SAMPLES_PER_PASS: {"k": 1}, TRANSITION_SHARD: {"rows": 256}, COND_SHARD: {"rows": 256}, LOGITS_SHARD: {"rows": 256}, TRIMUL_CHUNK: {"rows": 512}}   # the kit's defaults (flags override: ctx.setting)
TRACED = {SAMPLES_PER_PASS: 0, TRANSITION_SHARD: 0, COND_SHARD: 0, LOGITS_SHARD: 0, TRIMUL_CHUNK: 0}   # trace-time counters of the patched sites (the launcher records them at exit)
OBSERVED: dict = {}                                                         # values seen at trace time (stock's shard spec, num_samples), recorded at exit
_STOCK: dict = {}                                                           # the un-levered attributes, for undo


def _import(name: str):
    import importlib
    return importlib.import_module(name)


def _positive_int(v):
    n = int(v)
    if n < 1:
        raise ValueError(f"{n} < 1")
    return n


# ------------------------------------------------------------------------------------------------------------ samples_per_pass




def noise_draw():
    """The sampler's normal draw: ``jax.random.normal`` — the fork's own draw at the padded shape, which is the stock route's (every mode pads
    kernel_tile). It is on the record (OBSERVED samples_per_pass.noise_draw)."""
    import jax
    return jax.random.normal


def _scope_of(sample_fn):
    """The scope context a tree lever attached to the ``diffusion_head.sample`` this factory wraps (``_sampler_bf16_scope`` on the function or
    on any ``__wrapped_stock__`` / ``_sampler_bf16_inner`` beneath it), else a null context: a sampler that re-implements the body enters it as the wrapper would."""
    import contextlib
    f, seen = sample_fn, 0
    while f is not None and seen < 8:
        ctx = getattr(f, "_sampler_bf16_scope", None)
        if ctx is not None:
            return ctx
        f, seen = (getattr(f, "__wrapped_stock__", None) or getattr(f, "_sampler_bf16_inner", None)), seen + 1
    return contextlib.nullcontext


def chunked_sample_factory(stock_sample, k: int, DH):
    """The fork's ``diffusion_head.sample`` with the vmapped 200-step scan run per chunk of ``k`` samples. Draws as stock: one
    ``split(key)`` for the initial noise, one ``normal`` of the full ``(num_samples, …)`` shape, ``split(key, num_samples)`` per-sample keys —
    sliced per chunk (ckpt.sample_chunks, stock order); the per-step body is the fork's own (random_augmentation, the schedule, the
    denoiser call). ``k >= num_samples`` runs the stock sampler unchanged (noted)."""
    from opt_core.mem.ckpt import sample_chunks

    def sample(denoising_step, batch, key, config):
        import haiku as hk
        import jax
        import jax.numpy as jnp
        num_samples = int(config.num_samples)
        OBSERVED.setdefault(SAMPLES_PER_PASS, {}).update({"num_samples": num_samples, "k": k})
        if k >= num_samples:
            OBSERVED[SAMPLES_PER_PASS]["stock_path"] = True
            return stock_sample(denoising_step=denoising_step, batch=batch, key=key, config=config)
        TRACED[SAMPLES_PER_PASS] += 1
        mask = batch.predicted_structure_info.atom_mask
        lever_scope = _scope_of(stock_sample)                                # a tree lever that scopes ``sample`` (SAMPLER_BF16): this body re-implements it, so it enters that scope itself

        def apply_denoising_step(carry, noise_level):                       # the fork's body (diffusion_head.py:381-409)
            key, positions, noise_level_prev = carry
            key, key_noise, key_aug = jax.random.split(key, 3)
            positions = DH.random_augmentation(rng_key=key_aug, positions=positions, mask=mask)
            gamma = config.gamma_0 * (noise_level > config.gamma_min)
            t_hat = noise_level_prev * (1 + gamma)
            noise_scale = config.noise_scale * jnp.sqrt(jnp.maximum(t_hat**2 - noise_level_prev**2, 0.0))
            noise = noise_scale * normal(key_noise, positions.shape)
            positions_noisy = positions + noise
            positions_denoised = denoising_step(positions_noisy, t_hat)
            grad = (positions_noisy - positions_denoised) / t_hat
            d_t = noise_level - t_hat
            positions_out = positions_noisy + config.step_scale * d_t * grad
            return (key, positions_out, noise_level), positions_out

        normal = noise_draw()                                              # the fork's own jax.random.normal at the padded shape
        OBSERVED[SAMPLES_PER_PASS]["noise_draw"] = getattr(normal, "__module__", "?") + "." + getattr(normal, "__name__", "?")
        noise_levels = DH.noise_schedule(jnp.linspace(0, 1, config.steps + 1))
        key, noise_key = jax.random.split(key)
        positions = normal(noise_key, (num_samples,) + mask.shape + (3,))
        positions *= noise_levels[0]
        keys = jax.random.split(key, num_samples)
        outs = []
        with lever_scope():
            for sl in sample_chunks(num_samples, k):
                n = sl.stop - sl.start
                init = (keys[sl], positions[sl], jnp.tile(noise_levels[None, 0], (n,)))
                step = hk.vmap(apply_denoising_step, in_axes=(0, None), split_rng=(not hk.running_init()))
                result, _ = hk.scan(step, init, noise_levels[1:], unroll=4)
                outs.append(result[1])
        positions_out = jnp.concatenate(outs, axis=0)
        final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))
        return {"atom_positions": positions_out, "mask": final_dense_atom_mask}

    sample.__wrapped_stock__ = stock_sample
    return sample


def _applies_samples_per_pass(ctx) -> Optional[registry.Refusal]:
    h = ctx.require(SAMPLES_PER_PASS, "module", "attr")
    try:
        DH = _import(h["module"])
    except ImportError as e:
        return refuse(SAMPLES_PER_PASS, "hooks.module", f"{h['module']} not importable: {e}")
    fn = getattr(DH, h["attr"], None)
    if fn is None:
        return refuse(SAMPLES_PER_PASS, "hooks.attr", f"{h['module']}.{h['attr']} is absent")
    if getattr(fn, "__wrapped_stock__", None) is not None:
        return refuse(SAMPLES_PER_PASS, "site", f"{h['module']}.{h['attr']} is already the chunked sampler (applied twice)")
    for name in ("random_augmentation", "noise_schedule"):
        if not callable(getattr(DH, name, None)):
            return refuse(SAMPLES_PER_PASS, "hooks.module", f"{h['module']}.{name} is absent (the sampler body calls it)")
    try:
        ctx.setting(SAMPLES_PER_PASS, "k", SETTINGS[SAMPLES_PER_PASS]["k"], cast=_positive_int)
    except registry.RefusalError as e:
        return e.refusal
    return None


@registry.register(SAMPLES_PER_PASS, family="ckpt", exact="measured",
                   exact_reason="stock draw order kept (the per-sample keys and the one initial-noise draw are stock's, sliced per chunk); the "
                                "denoiser runs on a smaller batch, so XLA's fusion and kernel choice may differ: bitwise where "
                                "measured equal, inside the band otherwise (B2)",
                   applies=_applies_samples_per_pass, description="the vmapped diffusion sampler run k samples per pass, stock draws kept",
                   preconditions=("hooks.module", "hooks.attr", "stock_source", "site"), settings=("k",), frameworks=(FRAMEWORK,),
                   replace=True)                                            # replaces the core's torch lever of this name (opt_core.mem.ckpt.samples_per_pass: torch RNG discipline) with the JAX transcription
def samples_per_pass(ctx) -> Applied:
    h = ctx.hooks[SAMPLES_PER_PASS]
    k = ctx.setting(SAMPLES_PER_PASS, "k", SETTINGS[SAMPLES_PER_PASS]["k"], cast=_positive_int)
    DH = _import(h["module"])
    stock = getattr(DH, h["attr"])
    _STOCK[SAMPLES_PER_PASS] = (DH, h["attr"], stock)
    setattr(DH, h["attr"], chunked_sample_factory(stock, k, DH))

    def undo():
        setattr(DH, h["attr"], stock)

    site = f"{h['module']}.{h['attr']}"
    if ctx.record is not None:
        ctx.record.mark(SAMPLES_PER_PASS, detail=f"{site} rebound (k={k})")
    return Applied(lever=SAMPLES_PER_PASS, settings={"k": k, "draw_order": "stock"}, sites=(site,), undo=undo)


# ------------------------------------------------------------------------------------------------------------ transition_shard


def _applies_transition_shard(ctx) -> Optional[registry.Refusal]:
    h = ctx.require(TRANSITION_SHARD, "module", "cls", "path")
    try:
        M = _import(h["module"])
    except ImportError as e:
        return refuse(TRANSITION_SHARD, "hooks.module", f"{h['module']} not importable: {e}")
    cls = getattr(M, h["cls"], None)
    if cls is None or not callable(getattr(cls, "__init__", None)):
        return refuse(TRANSITION_SHARD, "hooks.cls", f"{h['module']}.{h['cls']} is absent")
    if getattr(cls.__init__, "__big_transition_shard__", False):
        return refuse(TRANSITION_SHARD, "site", f"{h['module']}.{h['cls']}.__init__ is already wrapped (applied twice)")
    cfg_cls = getattr(cls, "Config", None)
    try:
        from opt_core.mem.jax_mem import get_config_path
        get_config_path(cfg_cls(), h["path"])                                  # the knob exists on a default config (nothing is invented)
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        return refuse(TRANSITION_SHARD, "hooks.path", f"{h['cls']}.Config().{h['path']} unreadable: {e}")
    try:
        ctx.setting(TRANSITION_SHARD, "rows", SETTINGS[TRANSITION_SHARD]["rows"], cast=_positive_int)
    except registry.RefusalError as e:
        return e.refusal
    return None


@registry.register(TRANSITION_SHARD, family="jax", exact="measured",
                   exact_reason="a sub-batch of the pair transition: a per-position MLP whose per-row arithmetic does not depend on the row "
                                "count, but the per-shard kernels may differ from the full-width ones under XLA (opt_core.mem.jax_mem subbatch): "
                                "bitwise where measured equal, band otherwise",
                   applies=_applies_transition_shard, description="GlobalConfig.pair_transition_shard_spec = ((None, rows),): the pair transitions in row shards at every size",
                   preconditions=("hooks.module", "hooks.cls", "hooks.path", "site"), settings=("rows",), frameworks=(FRAMEWORK,))
def transition_shard(ctx) -> Applied:
    from opt_core.mem.jax_mem import set_config_path
    h = ctx.hooks[TRANSITION_SHARD]
    rows = ctx.setting(TRANSITION_SHARD, "rows", SETTINGS[TRANSITION_SHARD]["rows"], cast=_positive_int)
    M = _import(h["module"]); cls = getattr(M, h["cls"]); stock_init = cls.__init__
    spec = ((None, rows),)

    def __init__(self, config, *args, **kwargs):
        rec = set_config_path(config, h["path"], spec)                         # the library's transform: sets the knob, returns stock's value beside it
        TRACED[TRANSITION_SHARD] += 1
        OBSERVED.setdefault(TRANSITION_SHARD, {}).update({"stock": repr(rec["stock_value"]), "value": repr(rec["value"]), "path": rec["path"]})
        return stock_init(self, config, *args, **kwargs)

    __init__.__big_transition_shard__ = True
    __init__.__wrapped__ = stock_init
    _STOCK[TRANSITION_SHARD] = (cls, stock_init)
    cls.__init__ = __init__

    def undo():
        cls.__init__ = stock_init

    site = f"{h['module']}.{h['cls']}.__init__ -> config.{h['path']}"
    if ctx.record is not None:
        ctx.record.mark(TRANSITION_SHARD, detail=f"{site} = {spec!r}")
    return Applied(lever=TRANSITION_SHARD, settings={"rows": rows, "spec": spec, "path": h["path"]}, sites=(site,), undo=undo)


# ------------------------------------------------------------------------------------------------------------ cond_shard


def relative_encoding_rows(rows: dict, cols: dict, max_relative_idx: int, max_relative_chain: int):
    """The fork's ``featurization.create_relative_encoding`` (featurization.py:177-257) for one row shard: ``rows`` are the shard's per-token
    arrays (token_index, residue_index, asym_id, entity_id, sym_id), ``cols`` the full ones — the same ops, left = rows, right = cols."""
    import jax
    import jax.numpy as jnp
    rel_feats = []
    left_asym_id, right_asym_id = rows["asym_id"][:, None], cols["asym_id"][None, :]
    left_residue_index, right_residue_index = rows["residue_index"][:, None], cols["residue_index"][None, :]
    left_token_index, right_token_index = rows["token_index"][:, None], cols["token_index"][None, :]
    left_entity_id, right_entity_id = rows["entity_id"][:, None], cols["entity_id"][None, :]
    left_sym_id, right_sym_id = rows["sym_id"][:, None], cols["sym_id"][None, :]
    offset = left_residue_index - right_residue_index
    clipped_offset = jnp.clip(offset + max_relative_idx, min=0, max=2 * max_relative_idx)
    asym_id_same = left_asym_id == right_asym_id
    final_offset = jnp.where(asym_id_same, clipped_offset, (2 * max_relative_idx + 1) * jnp.ones_like(clipped_offset))
    rel_pos = jax.nn.one_hot(final_offset, 2 * max_relative_idx + 2)
    rel_feats.append(rel_pos)
    token_offset = left_token_index - right_token_index
    clipped_token_offset = jnp.clip(token_offset + max_relative_idx, min=0, max=2 * max_relative_idx)
    residue_same = (left_asym_id == right_asym_id) & (left_residue_index == right_residue_index)
    final_token_offset = jnp.where(residue_same, clipped_token_offset, (2 * max_relative_idx + 1) * jnp.ones_like(clipped_token_offset))
    rel_token = jax.nn.one_hot(final_token_offset, 2 * max_relative_idx + 2)
    rel_feats.append(rel_token)
    entity_id_same = left_entity_id == right_entity_id
    rel_feats.append(entity_id_same.astype(rel_pos.dtype)[..., None])
    rel_sym_id = left_sym_id - right_sym_id
    max_rel_chain = max_relative_chain
    clipped_rel_chain = jnp.clip(rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain)
    final_rel_chain = jnp.where(entity_id_same, clipped_rel_chain, (2 * max_rel_chain + 1) * jnp.ones_like(clipped_rel_chain))
    rel_chain = jax.nn.one_hot(final_rel_chain, 2 * max_relative_chain + 2)
    rel_feats.append(rel_chain)
    return jnp.concatenate(rel_feats, axis=-1)


REL_FIELDS = ("token_index", "residue_index", "asym_id", "entity_id", "sym_id")   # the token features create_relative_encoding reads


def pair_conditioning_rows(pe, rid, cols, use_conditioning, *, pair_channel: int, global_config, hm, DT):
    """One row block of the fork's pair conditioning (diffusion_head.py:152-176; the hoist's ``_pair_conditioning`` is the same ops):
    ``pe`` = the block's rows of the trunk pair embedding [rows, N, C], ``rid`` = their token indices [rows], ``cols`` = the full per-token
    features (REL_FIELDS). The conditioning gate, the relative encoding for the block (:func:`relative_encoding_rows`), the
    concatenation, the LayerNorm, the initial projection and the two pair transitions — the stock Haiku module names (``hm`` = the
    fork's haiku_modules, ``DT`` = diffusion_transformer), so the parameters are stock's. Every op is position-wise in the rows: the
    block's result is stock's rows. THE transcription (B21): the single-card lever maps it with the fork's
    ``mapping.sharded_apply``; the multi-card shard composes the same function under ``shard_map``."""
    import jax.numpy as jnp
    pair_embedding = use_conditioning * pe
    rel_features = relative_encoding_rows({k: v[rid] for k, v in cols.items()}, cols, 32, 2).astype(pair_embedding.dtype)
    features_2d = jnp.concatenate([pair_embedding, rel_features], axis=-1)
    pair_cond = hm.Linear(pair_channel, precision="highest", name="pair_cond_initial_projection")(
        hm.LayerNorm(use_fast_variance=False, create_offset=False, name="pair_cond_initial_norm")(features_2d))
    for idx in range(2):
        pair_cond += DT.transition_block(pair_cond, 2, global_config, name=f"pair_transition_{idx}")
    return pair_cond


def pair_conditioning_sharded_factory(rows: int, DH, MP, DT):
    """The pair half of the fork's ``DiffusionHead._conditioning`` computed in ``rows``-row blocks (:func:`pair_conditioning_rows`) through
    the fork's ``mapping.sharded_apply``. Nothing of pair size but the input plane and the output plane is live outside a block."""
    hm = DH.hm

    def pair_conditioning(self, batch, embeddings, use_conditioning):
        import jax.numpy as jnp
        pair = embeddings["pair"]
        tf = batch.token_features
        cols = {k: getattr(tf, k) for k in REL_FIELDS}
        n = int(pair.shape[0])
        TRACED[COND_SHARD] += 1
        OBSERVED.setdefault(COND_SHARD, {}).setdefault("shapes", []).append([int(d) for d in pair.shape])
        block = functools.partial(pair_conditioning_rows, cols=cols, use_conditioning=use_conditioning,
                                  pair_channel=self.config.conditioning.pair_channel, global_config=self.global_config, hm=hm, DT=DT)
        row_ids = jnp.arange(n)
        if n <= rows:
            return block(pair, row_ids)
        return MP.sharded_apply(block, shard_size=rows)(pair, row_ids)

    return pair_conditioning


def conditioning_factory(pair_half, DH, DT):
    """The fork's ``DiffusionHead._conditioning`` (diffusion_head.py:144-258) around a given pair half — ``pair_half(self, batch, embeddings,
    use_conditioning) -> pair_cond`` — with the single half verbatim (the OF3 restype/profile re-insertion, the Fourier parameters, the noise
    embedding, the two single transitions). The single-card lever passes :func:`pair_conditioning_sharded_factory`'s row-blocked pair half
    (:func:`conditioning_sharded_factory`); the row-sharded pair stack (opt_core.mem.rowpair_jax.alphafold3, ``b21`` = this module) passes each
    device's row block of :func:`pair_conditioning_rows` under its mesh. One body either way."""
    import haiku as hk
    hm, residue_names, noise_level_embeddings = DH.hm, DH.residue_names, DH.noise_level_embeddings

    def _conditioning(self, batch, embeddings, noise_level, use_conditioning):
        import jax.numpy as jnp
        single_embedding = use_conditioning * embeddings["single"]
        pair_cond = pair_half(self, batch, embeddings, use_conditioning)
        target_feat = embeddings["target_feat"]
        features_1d = jnp.concatenate([single_embedding, target_feat], axis=-1)
        if self.global_config.of3_weights:
            num_af3_restypes = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
            pad = jnp.zeros_like(features_1d[..., :1])
            single_channels = single_embedding.shape[-1]
            aatype_end = single_channels + num_af3_restypes
            profile_end = aatype_end + num_af3_restypes
            features_1d = jnp.concatenate([features_1d[..., :aatype_end], pad, features_1d[..., aatype_end:profile_end], pad, features_1d[..., profile_end:]], axis=-1)
        single_cond = hm.LayerNorm(use_fast_variance=False, create_offset=False, name="single_cond_initial_norm")(features_1d)
        single_cond = hm.Linear(self.config.conditioning.seq_channel, precision="highest", name="single_cond_initial_projection")(single_cond)
        if self.global_config.of3_weights:
            _dim = len(noise_level_embeddings._WEIGHT)
            fourier_weight = hk.get_parameter("fourier_embedding_weight", shape=[_dim], dtype=jnp.float32, init=hk.initializers.Constant(0.0))
            fourier_bias = hk.get_parameter("fourier_embedding_bias", shape=[_dim], dtype=jnp.float32, init=hk.initializers.Constant(0.0))
            noise_embedding = noise_level_embeddings.noise_embeddings(sigma_scaled_noise_level=noise_level / DH.SIGMA_DATA, weight=fourier_weight, bias=fourier_bias)
        else:
            noise_embedding = noise_level_embeddings.noise_embeddings(sigma_scaled_noise_level=noise_level / DH.SIGMA_DATA)
        single_cond += hm.Linear(self.config.conditioning.seq_channel, precision="highest", name="noise_embedding_initial_projection")(
            hm.LayerNorm(use_fast_variance=False, create_offset=False, name="noise_embedding_initial_norm")(noise_embedding))
        for idx in range(2):
            single_cond += DT.transition_block(single_cond, 2, self.global_config, name=f"single_transition_{idx}")
        return single_cond, pair_cond

    return _conditioning


def conditioning_sharded_factory(rows: int, DH, MP, DT):
    """The fork's ``DiffusionHead._conditioning`` with the pair half computed in ``rows``-row blocks (:func:`pair_conditioning_sharded_factory`)."""
    return conditioning_factory(pair_conditioning_sharded_factory(rows, DH, MP, DT), DH, DT)


def _applies_cond_shard(ctx) -> Optional[registry.Refusal]:
    import sys
    h = ctx.require(COND_SHARD, "module", "cls", "method", "mapping", "hoist_module", "hoist_attr", "rel_module", "rel_attr")
    try:
        DH, MP = _import(h["module"]), _import(h["mapping"])
        DT = _import("alphafold3.model.network.diffusion_transformer")
    except ImportError as e:
        return refuse(COND_SHARD, "hooks.module", f"a hook module is not importable: {e}")
    cls = getattr(DH, h["cls"], None)
    meth = getattr(cls, h["method"], None) if cls is not None else None
    if meth is None:
        return refuse(COND_SHARD, "hooks.cls", f"{h['module']}.{h['cls']}.{h['method']} is absent")
    if getattr(meth, "__big_cond_shard__", False):
        return refuse(COND_SHARD, "site", f"{h['module']}.{h['cls']}.{h['method']} is already the sharded conditioning (applied twice)")
    for name in ("hm", "residue_names", "noise_level_embeddings", "SIGMA_DATA"):
        if not hasattr(DH, name):
            return refuse(COND_SHARD, "hooks.module", f"{h['module']}.{name} is absent (the transcription reads it)")
    try:
        FZ = _import(h["rel_module"])
    except ImportError as e:
        return refuse(COND_SHARD, "hooks.module", f"{h['rel_module']} not importable: {e}")
    if not hasattr(FZ, h["rel_attr"]):
        return refuse(COND_SHARD, "rel_source", f"{h['rel_module']}.{h['rel_attr']} is absent")
    if not callable(getattr(MP, "sharded_apply", None)) or not callable(getattr(DT, "transition_block", None)):
        return refuse(COND_SHARD, "hooks.mapping", f"{h['mapping']}.sharded_apply or diffusion_transformer.transition_block is absent")
    hoist = sys.modules.get(h["hoist_module"])
    if hoist is not None:
        fn = getattr(hoist, h["hoist_attr"], None)
        if fn is None:
            return refuse(COND_SHARD, "hoist_site", f"{h['hoist_module']}.{h['hoist_attr']} is absent (the loaded hoist is not this tree's add-on)")
    try:
        ctx.setting(COND_SHARD, "rows", SETTINGS[COND_SHARD]["rows"], cast=_positive_int)
    except registry.RefusalError as e:
        return e.refusal
    return None


@registry.register(COND_SHARD, family="chunk", exact="measured",
                   exact_reason="the pair conditioning in row shards through the fork's own mapping.sharded_apply (the pairformer's form): every op "
                                "is position-wise and no reduction is split, but the per-shard kernels may differ from the full-plane ones under "
                                "XLA — bitwise where measured equal, band otherwise",
                   applies=_applies_cond_shard, description="the diffusion head's pair conditioning (concat, norm, projection, two transitions) in row shards",
                   preconditions=("hooks.module", "hooks.cls", "hooks.mapping", "stock_source", "rel_source", "site", "hoist_site", "hoist_source"), settings=("rows",),
                   frameworks=(FRAMEWORK,))
def cond_shard(ctx) -> Applied:
    import sys
    h = ctx.hooks[COND_SHARD]
    rows = ctx.setting(COND_SHARD, "rows", SETTINGS[COND_SHARD]["rows"], cast=_positive_int)
    DH, MP, DT = _import(h["module"]), _import(h["mapping"]), _import("alphafold3.model.network.diffusion_transformer")
    cls = getattr(DH, h["cls"]); stock = getattr(cls, h["method"])
    new = conditioning_sharded_factory(rows, DH, MP, DT)
    new.__big_cond_shard__ = True
    new.__wrapped__ = stock
    setattr(cls, h["method"], new)
    sites = [f"{h['module']}.{h['cls']}.{h['method']}"]
    undos = [lambda: setattr(cls, h["method"], stock)]
    hoist = sys.modules.get(h["hoist_module"])
    if hoist is not None:
        hstock = getattr(hoist, h["hoist_attr"])
        hnew = pair_conditioning_sharded_factory(rows, DH, MP, DT)
        hnew.__wrapped__ = hstock
        setattr(hoist, h["hoist_attr"], hnew)
        sites.append(f"{h['hoist_module']}.{h['hoist_attr']}")
        undos.append(lambda: setattr(hoist, h["hoist_attr"], hstock))
    _STOCK[COND_SHARD] = (cls, h["method"], stock)

    def undo():
        for u in undos:
            u()

    if ctx.record is not None:
        ctx.record.mark(COND_SHARD, detail=f"{', '.join(sites)} in {rows}-row shards")
    return Applied(lever=COND_SHARD, settings={"rows": rows}, sites=tuple(sites), undo=undo)


def pair_logits_blocks(pair_cond, rows: int, *, num_head: int, hm, jnp):
    """The OF3 transformer block's pair path (diffusion_transformer.py:223-231) in ``rows``-row blocks: ONE LayerNorm(pair_input_layer_norm)
    and ONE Linear(pair_logits_projection) module (the stock names; the block's own stacked parameters) applied to each row block of
    ``pair_cond`` [N, N, C] in turn, the blocks' logits [rows, N, H] concatenated to [N, N, H]. A plain unrolled loop, not
    ``mapping.sharded_apply``: inside ``hk.experimental.layer_stack`` the fork's sharded_apply (``hk.eval_shape``) cannot see the
    stack's parameter getter (apply fails with 'Unable to retrieve parameter'); the loop creates the modules once, in the block's scope."""
    ln = hm.LayerNorm(name="pair_input_layer_norm", use_fast_variance=False, create_offset=False)
    lin = hm.Linear(num_head, name="pair_logits_projection")
    n = int(pair_cond.shape[0])
    if n <= rows:
        return lin(ln(pair_cond))
    outs = [lin(ln(pair_cond[i:i + rows])) for i in range(0, n, rows)]
    return jnp.concatenate(outs, axis=0)


def transformer_class_factory(rows: int, DT):
    """A subclass of the fork's ``Transformer`` whose ``__call__`` (diffusion_transformer.py:209-295) computes the OF3 block's pair path in
    ``rows``-row blocks (:func:`pair_logits_blocks`): the per-block f32 [N, N, C] ``pair_act`` is never materialised — the block's pair
    logits [N, N, H] are the only pair-sized value of the block. The AF3-weights branch is verbatim. A SUBCLASS, not a rebound method:
    Haiku wraps a module's methods at class creation (the wrapper enters the module's name scope, 'diffusion_head/transformer/'); a
    function assigned to the class afterwards runs outside that scope and its modules' parameters are not found at apply
    ('Unable to retrieve parameter scale for module pair_input_layer_norm'). The add-on's HoistTransformer
    takes the same route."""
    hk, jnp, hm = DT.hk, DT.jnp, DT.hm

    def __call__(self, act, mask, single_cond, pair_cond):
        assert self.config.num_blocks % self.config.super_block_size == 0
        num_super_blocks = self.config.num_blocks // self.config.super_block_size

        if self.global_config.of3_weights and pair_cond is not None:
            TRACED[LOGITS_SHARD] += 1
            OBSERVED.setdefault(LOGITS_SHARD, {}).setdefault("shapes", []).append([int(d) for d in pair_cond.shape])
            def block(act):  # pylint: disable=function-redefined
                block_pair_logits = pair_logits_blocks(pair_cond, rows, num_head=self.config.attention.num_head, hm=hm, jnp=jnp)
                block_pair_logits = jnp.transpose(block_pair_logits, [2, 0, 1])
                act += DT.self_attention(
                    act, mask, block_pair_logits,
                    self.config.attention, self.global_config, single_cond,
                    name=self.name,
                )
                act += DT.transition_block(
                    act, self.config.num_intermediate_factor,
                    self.global_config, single_cond, name=self.name,
                )
                return act

            def super_block(act):  # pylint: disable=function-redefined
                return hk.experimental.layer_stack(self.config.super_block_size)(block)(act)

            return hk.experimental.layer_stack(num_super_blocks)(super_block)(act)

        # Original AF3 mode: single shared pair LayerNorm precomputed before all blocks.
        def block(act, pair_logits):
            act += DT.self_attention(
                act,
                mask,
                pair_logits,
                self.config.attention,
                self.global_config,
                single_cond,
                name=self.name,
            )
            act += DT.transition_block(
                act,
                self.config.num_intermediate_factor,
                self.global_config,
                single_cond,
                name=self.name,
            )
            return act, None

        # Precompute pair logits for performance
        if pair_cond is None:
            pair_act = None
        else:
            pair_act = hm.LayerNorm(
                name="pair_input_layer_norm",
                use_fast_variance=False,
                create_offset=False,
            )(pair_cond)

        def super_block(act):
            if pair_act is None:
                pair_logits = None
            else:
                pair_logits = hm.Linear(
                    (self.config.super_block_size, self.config.attention.num_head),
                    name="pair_logits_projection",
                )(pair_act)
                pair_logits = jnp.transpose(pair_logits, [2, 3, 0, 1])
            return hk.experimental.layer_stack(
                self.config.super_block_size, with_per_layer_inputs=True
            )(block)(act, pair_logits)

        return hk.experimental.layer_stack(
            num_super_blocks, with_per_layer_inputs=True
        )(super_block)(act)[0]

    cls = type("Transformer", (DT.Transformer,), {"__call__": __call__, "__doc__": DT.Transformer.__doc__, "__module__": DT.Transformer.__module__})
    cls.__big_logits_shard__ = True
    return cls


def _applies_logits_shard(ctx) -> Optional[registry.Refusal]:
    h = ctx.require(LOGITS_SHARD, "module", "cls", "method", "mapping")
    try:
        DT, MP = _import(h["module"]), _import(h["mapping"])
    except ImportError as e:
        return refuse(LOGITS_SHARD, "hooks.module", f"a hook module is not importable: {e}")
    cls = getattr(DT, h["cls"], None)
    meth = getattr(cls, h["method"], None) if cls is not None else None
    if meth is None:
        return refuse(LOGITS_SHARD, "hooks.cls", f"{h['module']}.{h['cls']}.{h['method']} is absent")
    if getattr(cls, "__big_logits_shard__", False):
        return refuse(LOGITS_SHARD, "site", f"{h['module']}.{h['cls']} is already the sharded pair path (applied twice)")
    if cls.__name__ != h["cls"] or cls.__module__ != h["module"]:
        return refuse(LOGITS_SHARD, "site", f"{h['module']}.{h['cls']} is {cls.__module__}.{cls.__name__} (another install owns the transformer)")
    for name in ("hk", "jnp", "hm", "self_attention", "transition_block"):
        if not hasattr(DT, name):
            return refuse(LOGITS_SHARD, "hooks.module", f"{h['module']}.{name} is absent (the transcription reads it)")
    try:
        ctx.setting(LOGITS_SHARD, "rows", SETTINGS[LOGITS_SHARD]["rows"], cast=_positive_int)
    except registry.RefusalError as e:
        return e.refusal
    return None


@registry.register(LOGITS_SHARD, family="chunk", exact="measured",
                   exact_reason="the OF3 transformer block's pair path (LayerNorm + logits projection, per block) in row blocks (an unrolled loop "
                                "over row slices, the modules created once): position-wise ops, no reduction split; the per-block kernels may differ "
                                "from the full-plane ones under XLA — bitwise where measured equal, band otherwise",
                   applies=_applies_logits_shard, description="the diffusion transformer's per-block pair LayerNorm + logits projection in row shards",
                   preconditions=("hooks.module", "hooks.cls", "hooks.mapping", "stock_source", "site"), settings=("rows",), frameworks=(FRAMEWORK,))
def logits_shard(ctx) -> Applied:
    h = ctx.hooks[LOGITS_SHARD]
    rows = ctx.setting(LOGITS_SHARD, "rows", SETTINGS[LOGITS_SHARD]["rows"], cast=_positive_int)
    DT = _import(h["module"])
    stock_cls = getattr(DT, h["cls"])
    new_cls = transformer_class_factory(rows, DT)
    new_cls.__wrapped__ = stock_cls
    setattr(DT, h["cls"], new_cls)                                          # DiffusionHead looks the class up on the module at call time
    _STOCK[LOGITS_SHARD] = (DT, h["cls"], stock_cls)
    site = f"{h['module']}.{h['cls']} (subclass: {h['method']})"
    if ctx.record is not None:
        ctx.record.mark(LOGITS_SHARD, detail=f"{site} in {rows}-row blocks (the OF3 block's pair path)")
    return Applied(lever=LOGITS_SHARD, settings={"rows": rows}, sites=(site,), undo=lambda: setattr(DT, h["cls"], stock_cls))


def trimul_class_factory(rows: int, MODS, base_cls, glut=None):
    """A subclass of the INSTALLED ``TriangleMultiplication`` (``base_cls``: the fork's class, or the exact line's GLUT subclass — the same
    stock body with the gated projection fused, af3_pallas_levers.make_patched_trimul_class) whose ``__call__`` (modules.py:258-344)
    computes the module in ``rows``-row blocks of the output: the gated projection's contracted half (``b`` for the outgoing equation
    'ikc,jkc->ijc', ``a`` for the incoming 'kjc,kic->ijc') is computed ONCE for the full plane, [C, N, N] bf16, by row blocks; then per
    output row block the other half is recomputed for that block (rows of the plane for outgoing, columns for incoming), the einsum,
    center_norm, output_projection and the gate applied to the block, the blocks concatenated. Stock materialises the full gated
    projection [2C, N, N] (17.8 GiB at 6114), its transposes and the einsum output at once (the 6114-token peak). The same
    contraction per (i, j); the modules are created once (the stock Haiku names, so the parameters are stock's); the gated projection
    is the base's own route — ``glut`` = the exact line's fused transposed+masked Pallas GLU when that class is installed, else the
    fork's ``tokamax.gated_linear_unit`` + transpose + mask. The N <= rows case is the base's body in these helpers' terms."""
    hk, jax, jnp, hm, tokamax = MODS.hk, MODS.jax, MODS.jnp, MODS.hm, MODS.tokamax

    def __call__(self, act, mask):
        mask_b = mask[None, ...]
        num_channels = act.shape[-1]
        n = int(act.shape[0])
        equation = {
            'ikc,jkc->ijc': 'cik,cjk->cij',
            'kjc,kic->ijc': 'ckj,cki->cij',
        }[self.config.equation]
        outgoing = self.config.equation == 'ikc,jkc->ijc'
        act = hm.LayerNorm(name='left_norm_input')(act)
        input_act = act
        if self.config.use_glu_kernel:
            weights_projection, _ = hm.haiku_linear_get_params(
                act, num_output=num_channels * 2, name='projection'
            )
            weights_gate, _ = hm.haiku_linear_get_params(
                act,
                num_output=num_channels * 2,
                initializer=self.global_config.final_init,
                name='gate',
            )
            weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)
            if glut is not None:                                        # the exact line's fused kernel: transpose + mask inside (same dtype) or the stock op outside
                def gated(x, m):                                        # x [.., .., C], m [.., ..] -> masked [2C, .., ..]
                    if m.dtype == x.dtype:
                        return glut(x, weights_glu, m, activation=jax.nn.sigmoid)
                    return glut(x, weights_glu, None, activation=jax.nn.sigmoid) * m[None, ...]
            else:
                def gated(x, m):
                    p = tokamax.gated_linear_unit(x, weights_glu, activation=jax.nn.sigmoid)
                    return jnp.transpose(p, (2, 0, 1)) * m[None, ...]
        else:
            lin_projection = hm.Linear(num_channels * 2, name='projection')
            lin_gate = hm.Linear(
                num_channels * 2,
                name='gate',
                bias_init=1.0,
                initializer=self.global_config.final_init,
            )

            def gated(x, m):
                p = jnp.transpose(lin_projection(x), (2, 0, 1)) * m[None, ...]
                g = jnp.transpose(lin_gate(x), (2, 0, 1))
                return p * jax.nn.sigmoid(g)
        center_norm = hm.LayerNorm(name='center_norm', axis=0, param_axis=0)
        output_projection = hm.Linear(
            num_channels,
            initializer=self.global_config.final_init,
            name='output_projection',
        )
        gating_linear = hm.Linear(
            num_channels,
            name='gating_linear',
            bias_init=1.0,
            initializer=self.global_config.final_init,
        )

        def halves(p):                                                  # [2C, .., ..] masked projection -> (a, b) each [C, .., ..]
            p = p.reshape(num_channels, 2, *p.shape[1:])
            a, b = jnp.split(p, 2, axis=1)
            return jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)

        def finish(blk, rows_act):                                      # [C, r, N] einsum output -> [r, N, C] gated output rows
            blk = center_norm(blk)
            blk = jnp.transpose(blk, (1, 2, 0))
            blk = output_projection(blk)
            return blk * jax.nn.sigmoid(gating_linear(rows_act))

        TRACED[TRIMUL_CHUNK] += 1
        OBSERVED.setdefault(TRIMUL_CHUNK, {}).setdefault("shapes", []).append([int(d) for d in act.shape] + [self.config.equation])
        if n <= rows:                                                   # the base's body, in these helpers' terms
            a, b = halves(gated(act, mask))
            return finish(jnp.einsum(equation, a, b), input_act)
        starts = list(range(0, n, rows))
        if outgoing:                                                    # out[c,i,j] = sum_k a[c,i,k] b[c,j,k]: b for every row once, a per row block
            full = jnp.concatenate([halves(gated(act[i:i + rows], mask[i:i + rows, :]))[1] for i in starts], axis=1)   # b [C, N, N]
        else:                                                           # out[c,i,j] = sum_k a[c,k,j] b[c,k,i]: a for every row once, b per column block
            full = jnp.concatenate([halves(gated(act[i:i + rows], mask[i:i + rows, :]))[0] for i in starts], axis=1)   # a [C, N, N]
        outs = []
        for i in starts:
            if outgoing:
                a_i = halves(gated(act[i:i + rows], mask[i:i + rows, :]))[0]                   # [C, r, N]
                blk = jnp.einsum(equation, a_i, full)                                          # [C, r, N]
            else:
                b_i = halves(gated(act[:, i:i + rows, :], mask[:, i:i + rows]))[1]             # [C, N, r]: the projection at (k, i in block)
                blk = jnp.einsum(equation, full, b_i)                                          # [C, r, N]
            outs.append(finish(blk, input_act[i:i + rows]))
        return jnp.concatenate(outs, axis=0)

    cls = type("TriangleMultiplication", (base_cls,), {"__call__": __call__, "__doc__": base_cls.__doc__, "__module__": base_cls.__module__})
    cls.__big_trimul_chunk__ = True
    return cls


def _trimul_base(MODS, h):
    """The installed TriangleMultiplication and the route it composes on: (cls, glut_fn | None, reason | None). Accepted: the fork's own class
    (glut None) or the exact line's GLUT subclass (module ``glut_module``, its ``glut_attr`` the fused kernel; the factory's source sha pinned);
    anything else (the fpf add-on's class) refuses."""
    import sys
    cls = getattr(MODS, h["cls"], None)
    if cls is None:
        return None, None, f"{h['module']}.{h['cls']} is absent"
    stock = next((c for c in cls.__mro__ if c.__module__ == h["module"] and c.__name__ == h["cls"]), None)
    if stock is None:
        return None, None, f"{h['module']}.{h['cls']} is {cls.__module__}.{cls.__name__}, not a subclass of the fork's (another install owns the module)"
    if cls is stock:
        return cls, None, None
    if cls.__module__ != h["glut_module"] or len([c for c in cls.__mro__ if c.__name__ == h["cls"]]) != 2:
        return None, None, f"{h['module']}.{h['cls']} is {cls.__module__}.{cls.__name__} (another install owns the module: the fpf line's kernels)"
    mod = sys.modules.get(h["glut_module"])
    glut = getattr(mod, h["glut_attr"], None) if mod is not None else None
    if not callable(glut):
        return None, None, f"{h['glut_module']}.{h['glut_attr']} is absent (the GLUT class is installed without its kernel)"
    if not hasattr(mod, h["glut_factory"]):
        return None, None, f"{h['glut_module']}.{h['glut_factory']} is absent (the composed GLUT body's factory)"
    return cls, glut, None


def _applies_trimul_chunk(ctx) -> Optional[registry.Refusal]:
    h = ctx.require(TRIMUL_CHUNK, "module", "cls", "method", "glut_module", "glut_attr", "glut_factory")
    try:
        MODS = _import(h["module"])
    except ImportError as e:
        return refuse(TRIMUL_CHUNK, "hooks.module", f"a hook module is not importable: {e}")
    cls = getattr(MODS, h["cls"], None)
    if cls is None or getattr(cls, h["method"], None) is None:
        return refuse(TRIMUL_CHUNK, "hooks.cls", f"{h['module']}.{h['cls']}.{h['method']} is absent")
    if getattr(cls, "__big_trimul_chunk__", False):
        return refuse(TRIMUL_CHUNK, "site", f"{h['module']}.{h['cls']} is already the row-blocked module (applied twice)")
    base, glut, why = _trimul_base(MODS, h)
    if why:
        return refuse(TRIMUL_CHUNK, "site", why)
    for name in ("hk", "jax", "jnp", "hm", "tokamax"):
        if not hasattr(MODS, name):
            return refuse(TRIMUL_CHUNK, "hooks.module", f"{h['module']}.{name} is absent (the transcription reads it)")
    if not callable(getattr(MODS.hm, "haiku_linear_get_params", None)):
        return refuse(TRIMUL_CHUNK, "hooks.module", "haiku_modules.haiku_linear_get_params is absent (the GLU branch reads it)")
    try:
        ctx.setting(TRIMUL_CHUNK, "rows", SETTINGS[TRIMUL_CHUNK]["rows"], cast=_positive_int)
    except registry.RefusalError as e:
        return e.refusal
    return None


@registry.register(TRIMUL_CHUNK, family="chunk", exact="measured",
                   exact_reason="TriangleMultiplication in row blocks of the output (the contracted half of the gated projection computed once for "
                                "the plane, the other half + einsum + center_norm + output projection + gate per block): the same contraction "
                                "per (i, j); the per-block GEMM tiling may differ from the full-plane one under XLA — bitwise where "
                                "measured equal, inside the band otherwise",
                   applies=_applies_trimul_chunk, description="the trunk's TriangleMultiplication (pairformer, MSA stack, confidence pairformer) in row blocks",
                   preconditions=("hooks.module", "hooks.cls", "stock_source", "site"), settings=("rows",), frameworks=(FRAMEWORK,))
def trimul_chunk(ctx) -> Applied:
    h = ctx.hooks[TRIMUL_CHUNK]
    rows = ctx.setting(TRIMUL_CHUNK, "rows", SETTINGS[TRIMUL_CHUNK]["rows"], cast=_positive_int)
    MODS = _import(h["module"])
    base_cls, glut, why = _trimul_base(MODS, h)
    if why:
        raise registry.RefusalError(refuse(TRIMUL_CHUNK, "site", why))
    new_cls = trimul_class_factory(rows, MODS, base_cls, glut)
    new_cls.__wrapped__ = base_cls
    setattr(MODS, h["cls"], new_cls)                                        # every caller (PairFormerIteration, the MSA stack) looks the class up on the module at call time
    _STOCK[TRIMUL_CHUNK] = (MODS, h["cls"], base_cls)
    route = f"{h['glut_module']}.{h['glut_attr']}" if glut is not None else "tokamax.gated_linear_unit"
    site = f"{h['module']}.{h['cls']} (subclass of {base_cls.__module__}.{base_cls.__name__}: {h['method']})"
    if ctx.record is not None:
        ctx.record.mark(TRIMUL_CHUNK, detail=f"{site} in {rows}-row blocks; gated projection = {route}")
    OBSERVED.setdefault(TRIMUL_CHUNK, {}).update({"base_class": f"{base_cls.__module__}.{base_cls.__name__}", "gated_projection": route})
    return Applied(lever=TRIMUL_CHUNK, settings={"rows": rows}, sites=(site,), undo=lambda: setattr(MODS, h["cls"], base_cls))


def exit_notes() -> list:
    """What the trace-time counters say, for the record at exit (0 traces = the executable was loaded, never traced here)."""
    return [f"{lv}: traced {n} time(s)" + (f" observed={OBSERVED[lv]}" if lv in OBSERVED else "") for lv, n in TRACED.items()]
