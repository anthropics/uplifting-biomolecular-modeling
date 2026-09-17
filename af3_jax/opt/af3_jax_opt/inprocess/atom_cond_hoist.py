"""ATOM_COND_HOIST — the atom cross-attention encoder's step-invariant conditioning computed ONCE per sample call, outside the 200-step
sampler loop (kind=schedule; strategy LOCAL.step_invariant_hoist, the FlashPairformer FPF_HOIST idea applied to the atom path: stock
recomputes the atom pair conditioning once per unrolled scan iteration although it depends on nothing the step produces).

What stock does (alphafold3/model/network/atom_cross_attention.py ``atom_cross_att_encoder``, called by ``DiffusionHead.__call__`` at
every denoising step with ``token_atoms_act`` = the noisy positions): before anything touches the positions it builds, from the batch's
reference structure / per-atom features and the trunk's single + pair embeddings only, (1) the per-atom single conditioning in queries and
keys layout (``queries_single_cond`` [subsets, 32, 128], ``keys_single_cond`` [subsets, 128, 128]) and the masks, (2) the atom-pair
conditioning ``pair_act`` [subsets, 32, 128, 16]: row/col embeddings + the trunk pair conditioning gathered to atom pairs + reference
offsets / distances / validity embeddings + a 3-layer pair MLP, and — inside both atom transformers (encoder AND decoder,
``diffusion_transformer.CrossAttTransformer.__call__``) — (3) the per-block pair logits ``LayerNorm(pair_act) -> Linear(16 -> 3 blocks x 4
heads)``. (1)-(3) are identical at every step and for every sample.

What changes: ONLY when (1)-(3) are computed. With the FlashPairformer hoist installed (FPF_HOIST: ``HoistDiffusionHead``, whose 'step'
hands the encoder the once-computed trunk pair conditioning), this module's ``diffusion_head.sample`` wrapper runs, once per sample call and
outside ``hk.scan`` / ``hk.vmap``, this module's ``DiffusionHead.__call__`` in its ``atom_precompute`` mode (a mode of ``__call__`` so the
modules are created in the head's own Haiku scope: parameter names unchanged) = the stock encoder's lines for (1)-(2) verbatim + the two
transformers' lines for (3) verbatim, and stashes the result for the trace of that sample call; the rebound ``atom_cross_att_encoder`` /
``atom_cross_att_decoder`` (module attributes, resolved at call time by the hoist's step and by stock) take the stashed tensors and run only
the position-dependent lines (position embedding, the 3 + 3 atom-transformer blocks with the precomputed logits as per-layer inputs,
aggregation, position update). Arithmetic per element unchanged (same ops, dtypes, precision='highest' pins, the same
``utils.bfloat16_context``); bitwise identity with the lever off depends on XLA compiling the hoisted ops into the same fusions — a
property of the compiled program, not something this source guarantees. Memory: the stash keeps pair_act (subsets*32*128*16 f32) + 2 x
logits (3*subsets*4*32*128 f32) resident for the sample call: ~0.6 GB at 1216 tokens (29k atoms), linear in atoms.

Steps aside BY NAME: without FPF_HOIST's step (stock ``DiffusionHead.__call__``: the trunk pair conditioning is itself built inside the
step) -> ``aside=no_hoist_step``, everything stock; the trunk's own encoder call (``evoformer_conditioning``, no stash) and the confidence
path are untouched (``passed``). Installed BEFORE the add-on's import so ``HoistDiffusionHead`` subclasses this module's DiffusionHead
(fpf_launch TREE_LEVERS order).
Switch: ``AF3_JAX_ATOM_COND_HOIST=1``.
Prints: the launcher prints ``XLEVER ATOM_COND_HOIST state=on precomputed=<n> enc=<n> dec=<n>`` from ``report()`` -> {"installed",
"precomputed" (sample calls hoisted), "enc"/"dec" (encoder / decoder calls served from the stash at trace time), "passed", "aside"}.
"""
import os

ENV_SWITCH = "AF3_JAX_ATOM_COND_HOIST"
REBINDS = ("alphafold3.model.network.atom_cross_attention:atom_cross_att_encoder", "alphafold3.model.network.atom_cross_attention:atom_cross_att_decoder",
           "alphafold3.model.network.diffusion_head:sample", "alphafold3.model.network.diffusion_head:DiffusionHead")   # what install() rebinds — `sample` is WRAPPED (the precompute runs once, then the bound sampler), DiffusionHead is subclassed BEFORE the add-on derives HoistDiffusionHead from it (fpf_launch INSTALL ORDER; tests/test_install_order.py)
_STATE = {"installed": False, "precomputed": 0, "enc": 0, "dec": 0, "passed": 0, "aside": {}, "stock": {}}
_STASH = []                                                               # the invariants of the sample call being traced (pushed by the sample wrapper, read at trace time)


def wanted(environ=os.environ) -> bool:
    return environ.get(ENV_SWITCH, "") == "1"


def _aside(reason):
    _STATE["aside"][reason] = _STATE["aside"].get(reason, 0) + 1


def _make(ACA, DT, DH):
    import haiku as hk
    import jax
    import jax.numpy as jnp
    hm, atom_layout, utils = ACA.hm, ACA.atom_layout, ACA.utils
    stock_enc, stock_dec, stock_sample = ACA.atom_cross_att_encoder, ACA.atom_cross_att_decoder, DH.sample

    class HoistCrossAttTransformer(DT.CrossAttTransformer):
        """CrossAttTransformer with mode='precompute' (the per-block pair logits, verbatim) and mode='step' (the blocks on precomputed logits)."""

        def __call__(self, queries_act, queries_mask, queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_cond,
                     mode="stock", pair_logits_pre=None):
            if mode == "stock":
                return super().__call__(queries_act, queries_mask, queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_cond)
            if mode == "precompute":
                pair_act = hm.LayerNorm(name="pair_input_layer_norm", use_fast_variance=False, create_offset=False)(pair_cond)
                pair_logits = hm.Linear((self.config.num_blocks, self.config.attention.num_head), name="pair_logits_projection")(pair_act)
                return jnp.transpose(pair_logits, [3, 0, 4, 1, 2])
            assert mode == "step" and pair_logits_pre is not None

            def block(queries_act, pair_logits):
                keys_act = atom_layout.convert(queries_to_keys, queries_act, layout_axes=(-3, -2))
                queries_act += DT.cross_attention(x_q=queries_act, x_k=keys_act, mask_q=queries_mask, mask_k=keys_mask, config=self.config.attention,
                                                  global_config=self.global_config, pair_logits=pair_logits, single_cond_q=queries_single_cond,
                                                  single_cond_k=keys_single_cond, name=self.name)
                queries_act += DT.transition_block(queries_act, self.config.num_intermediate_factor, self.global_config, queries_single_cond, name=self.name)
                return queries_act, None

            return hk.experimental.layer_stack(self.config.num_blocks, with_per_layer_inputs=True)(block)(queries_act, pair_logits_pre)[0]

    def encoder_invariants(trunk_single_cond, trunk_pair_cond, config, global_config, batch, name):
        """atom_cross_att_encoder's step-invariant lines, verbatim (everything that does not read token_atoms_act), + both transformers' pair logits."""
        c = config
        token_atoms_single_cond, _ = ACA._per_atom_conditioning(config, batch, name, global_config)
        token_atoms_mask = batch.predicted_structure_info.atom_mask
        queries_single_cond = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, token_atoms_single_cond, layout_axes=(-3, -2))
        queries_mask = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, token_atoms_mask, layout_axes=(-2, -1))
        if trunk_single_cond is not None:
            trunk_single_cond = hm.Linear(c.per_atom_channels, precision="highest", initializer=global_config.final_init, name=f"{name}_embed_trunk_single_cond")(
                hm.LayerNorm(use_fast_variance=False, create_offset=False, name=f"{name}_lnorm_trunk_single_cond")(trunk_single_cond))
            queries_single_cond += atom_layout.convert(batch.atom_cross_att.tokens_to_queries, trunk_single_cond, layout_axes=(-2,))
        queries_single_cond = queries_single_cond * queries_mask[..., None]
        keys_single_cond = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(-3, -2))
        keys_mask = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_mask, layout_axes=(-2, -1))
        row_act = hm.Linear(c.per_atom_pair_channels, name=f"{name}_single_to_pair_cond_row")(jax.nn.relu(queries_single_cond))
        pair_cond_keys_input = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_single_cond, layout_axes=(-3, -2))
        col_act = hm.Linear(c.per_atom_pair_channels, name=f"{name}_single_to_pair_cond_col")(jax.nn.relu(pair_cond_keys_input))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]
        if trunk_pair_cond is not None:
            trunk_pair_cond = hm.Linear(c.per_atom_pair_channels, precision="highest", initializer=global_config.final_init, name=f"{name}_embed_trunk_pair_cond")(
                hm.LayerNorm(use_fast_variance=False, create_offset=False, name=f"{name}_lnorm_trunk_pair_cond")(trunk_pair_cond))
            num_tokens = trunk_pair_cond.shape[0]
            tokens_to_queries = batch.atom_cross_att.tokens_to_queries
            tokens_to_keys = batch.atom_cross_att.tokens_to_keys
            trunk_pair_to_atom_pair = atom_layout.GatherInfo(
                gather_idxs=(num_tokens * tokens_to_queries.gather_idxs[:, :, None] + tokens_to_keys.gather_idxs[:, None, :]),
                gather_mask=(tokens_to_queries.gather_mask[:, :, None] & tokens_to_keys.gather_mask[:, None, :]),
                input_shape=jnp.array((num_tokens, num_tokens)))
            pair_act += atom_layout.convert(trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2))
        queries_ref_pos = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, batch.ref_structure.positions, layout_axes=(-3, -2))
        queries_ref_space_uid = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, batch.ref_structure.ref_space_uid, layout_axes=(-2, -1))
        keys_ref_pos = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_ref_pos, layout_axes=(-3, -2))
        keys_ref_space_uid = atom_layout.convert(batch.atom_cross_att.queries_to_keys, queries_ref_space_uid, layout_axes=(-2, -1))
        offsets_valid = queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        if global_config.of3_weights:
            offsets_valid = offsets_valid & keys_mask[:, None, :].astype(jnp.bool_)
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]
        pair_act += (hm.Linear(c.per_atom_pair_channels, precision="highest", name=f"{name}_embed_pair_offsets")(offsets) * offsets_valid[:, :, :, None])
        sq_dists = jnp.sum(jnp.square(offsets), axis=-1)
        pair_act += (hm.Linear(c.per_atom_pair_channels, name=f"{name}_embed_pair_distances")(1.0 / (1 + sq_dists[:, :, :, None])) * offsets_valid[:, :, :, None])
        pair_act += hm.Linear(c.per_atom_pair_channels, name=f"{name}_embed_pair_offsets_valid")(offsets_valid[:, :, :, None].astype(jnp.float32))
        pair_act2 = hm.Linear(c.per_atom_pair_channels, initializer="relu", name=f"{name}_pair_mlp_1")(jax.nn.relu(pair_act))
        pair_act2 = hm.Linear(c.per_atom_pair_channels, initializer="relu", name=f"{name}_pair_mlp_2")(jax.nn.relu(pair_act2))
        pair_act += hm.Linear(c.per_atom_pair_channels, initializer=global_config.final_init, name=f"{name}_pair_mlp_3")(jax.nn.relu(pair_act2))
        enc_logits = HoistCrossAttTransformer(c.atom_transformer, global_config, name=f"{name}_atom_transformer_encoder")(
            None, queries_mask, batch.atom_cross_att.queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_act, mode="precompute")
        dec_logits = HoistCrossAttTransformer(c.atom_transformer, global_config, name=f"{name}_atom_transformer_decoder")(
            None, queries_mask, batch.atom_cross_att.queries_to_keys, keys_mask, queries_single_cond, keys_single_cond, pair_act, mode="precompute")
        return {"queries_single_cond": queries_single_cond, "queries_mask": queries_mask, "keys_single_cond": keys_single_cond, "keys_mask": keys_mask,
                "pair_cond": pair_act, "enc_logits": enc_logits, "dec_logits": dec_logits}

    def atom_cross_att_encoder(token_atoms_act, trunk_single_cond, trunk_pair_cond, config, global_config, batch, name):
        inv = _STASH[-1] if _STASH else None
        if inv is None or inv.get("name") != name:                       # outside a hoisted sample call (the trunk's evoformer_conditioning call, the memory mode's own sampler): stock
            _STATE["passed"] += 1
            return stock_enc(token_atoms_act=token_atoms_act, trunk_single_cond=trunk_single_cond, trunk_pair_cond=trunk_pair_cond, config=config,
                             global_config=global_config, batch=batch, name=name)
        _STATE["enc"] += 1
        c = config
        queries_single_cond, queries_mask, keys_single_cond, keys_mask, pair_act = (inv["queries_single_cond"], inv["queries_mask"], inv["keys_single_cond"],
                                                                                  inv["keys_mask"], inv["pair_cond"])
        token_atoms_mask = batch.predicted_structure_info.atom_mask
        if token_atoms_act is None:
            queries_act = queries_single_cond
        else:
            queries_act = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, token_atoms_act, layout_axes=(-3, -2))
            queries_act = hm.Linear(c.per_atom_channels, precision="highest", name=f"{name}_atom_positions_to_features")(queries_act)
            queries_act *= queries_mask[..., None]
            queries_act += queries_single_cond
        queries_act = HoistCrossAttTransformer(c.atom_transformer, global_config, name=f"{name}_atom_transformer_encoder")(
            queries_act=queries_act, queries_mask=queries_mask, queries_to_keys=batch.atom_cross_att.queries_to_keys, keys_mask=keys_mask,
            queries_single_cond=queries_single_cond, keys_single_cond=keys_single_cond, pair_cond=pair_act, mode="step", pair_logits_pre=inv["enc_logits"])
        queries_act *= queries_mask[..., None]
        skip_connection = queries_act
        queries_act = hm.Linear(c.per_token_channels, name=f"{name}_project_atom_features_for_aggr")(queries_act)
        token_atoms_act = atom_layout.convert(batch.atom_cross_att.queries_to_token_atoms, queries_act, layout_axes=(-3, -2))
        token_act = utils.mask_mean(token_atoms_mask[..., None], jax.nn.relu(token_atoms_act), axis=-2)
        return ACA.AtomCrossAttEncoderOutput(token_act=token_act, skip_connection=skip_connection, queries_mask=queries_mask, queries_single_cond=queries_single_cond,
                                          keys_mask=keys_mask, keys_single_cond=keys_single_cond, pair_cond=pair_act)

    def atom_cross_att_decoder(token_act, enc, config, global_config, batch, name):
        inv = _STASH[-1] if _STASH else None
        if inv is None or inv.get("name") != name:
            _STATE["passed"] += 1
            return stock_dec(token_act=token_act, enc=enc, config=config, global_config=global_config, batch=batch, name=name)
        _STATE["dec"] += 1
        c = config
        token_act = hm.Linear(c.per_atom_channels, name=f"{name}_project_token_features_for_broadcast")(token_act)
        num_token, max_atoms_per_token = batch.atom_cross_att.queries_to_token_atoms.shape
        token_atom_act = jnp.broadcast_to(token_act[:, None, :], (num_token, max_atoms_per_token, c.per_atom_channels))
        queries_act = atom_layout.convert(batch.atom_cross_att.token_atoms_to_queries, token_atom_act, layout_axes=(-3, -2))
        queries_act += enc.skip_connection
        queries_act *= enc.queries_mask[..., None]
        queries_act = HoistCrossAttTransformer(c.atom_transformer, global_config, name=f"{name}_atom_transformer_decoder")(
            queries_act=queries_act, queries_mask=enc.queries_mask, queries_to_keys=batch.atom_cross_att.queries_to_keys, keys_mask=enc.keys_mask,
            queries_single_cond=enc.queries_single_cond, keys_single_cond=enc.keys_single_cond, pair_cond=enc.pair_cond, mode="step", pair_logits_pre=inv["dec_logits"])
        queries_act *= enc.queries_mask[..., None]
        queries_act = hm.LayerNorm(use_fast_variance=False, create_offset=False, name=f"{name}_atom_features_layer_norm")(queries_act)
        queries_position_update = hm.Linear(3, initializer=global_config.final_init, precision="highest", name=f"{name}_atom_features_to_position_update")(queries_act)
        position_update = atom_layout.convert(batch.atom_cross_att.queries_to_token_atoms, queries_position_update, layout_axes=(-3, -2))
        return position_update

    class ACHDiffusionHead(DH.DiffusionHead):
        """DiffusionHead whose ``__call__`` has one extra keyword, ``atom_precompute``: given, the call returns the encoder invariants, built in the
        head's ``__call__`` scope (Haiku names modules created in any OTHER method under ``~<method>/`` — the parameters live under ``__call__``'s
        scope, so the precompute must be a mode of ``__call__``, as the FlashPairformer hoist's is). The add-on's HoistDiffusionHead subclasses it."""

        def __call__(self, positions_noisy, noise_level, batch, embeddings, use_conditioning, atom_precompute=None):
            if atom_precompute is None:
                return super().__call__(positions_noisy, noise_level, batch, embeddings, use_conditioning)
            with utils.bfloat16_context():
                inv = encoder_invariants(trunk_single_cond=embeddings["single"], trunk_pair_cond=atom_precompute["trunk_pair_cond"], config=self.config,
                                         global_config=self.global_config, batch=batch, name="diffusion")
            inv["name"] = "diffusion"
            return inv

    def sample(denoising_step, batch, key, config):
        """diffusion_head.sample wrapper: with the hoist's step as the denoiser, precompute the atom invariants once (outside the scan), stash
        them for this trace, run the wrapped sampler, unstash."""
        f = getattr(denoising_step, "func", None)                        # the hoist's denoiser = functools.partial(<the DiffusionHead module instance>, batch=, embeddings=, mode="step", pre=)
        dm = f if isinstance(f, ACHDiffusionHead) else getattr(f, "__self__", None)   # a module instance (callable) or a bound method of one
        kw = dict(getattr(denoising_step, "keywords", None) or {})
        pre = kw.get("pre")
        if not (isinstance(dm, ACHDiffusionHead) and kw.get("mode") == "step" and isinstance(pre, dict) and "pair_cond" in pre and "batch" in kw and "embeddings" in kw):
            _aside("no_hoist_step" if isinstance(dm, ACHDiffusionHead) else "denoiser_not_a_head_partial")
            return stock_sample(denoising_step=denoising_step, batch=batch, key=key, config=config)
        inv = ACHDiffusionHead.__call__(dm, None, None, kw["batch"], kw["embeddings"], kw.get("use_conditioning", True),
                                        atom_precompute={"trunk_pair_cond": pre["pair_cond"]})   # this class's __call__ on the instance: the head's own scope, whatever subclass dm is
        _STATE["precomputed"] += 1
        _STASH.append(inv)
        try:
            return stock_sample(denoising_step=denoising_step, batch=batch, key=key, config=config)
        finally:
            _STASH.pop()

    sample._atom_cond_hoist = True
    for attr in ("_cond_share",):                                       # carry the inner sampler's marks outward (evidence only)
        if hasattr(stock_sample, attr):
            setattr(sample, attr, getattr(stock_sample, attr))
    return ACHDiffusionHead, atom_cross_att_encoder, atom_cross_att_decoder, sample


def install() -> bool:
    if _STATE["installed"]:
        return True
    from alphafold3.model.network import atom_cross_attention as ACA
    from alphafold3.model.network import diffusion_transformer as DT
    from alphafold3.model.network import diffusion_head as DH
    _STATE["stock"] = {"enc": ACA.atom_cross_att_encoder, "dec": ACA.atom_cross_att_decoder, "sample": DH.sample, "head": DH.DiffusionHead}
    head, enc, dec, sample = _make(ACA, DT, DH)
    ACA.atom_cross_att_encoder, ACA.atom_cross_att_decoder, DH.sample, DH.DiffusionHead = enc, dec, sample, head
    _STATE["installed"] = True
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        from alphafold3.model.network import atom_cross_attention as ACA
        from alphafold3.model.network import diffusion_head as DH
        ACA.atom_cross_att_encoder, ACA.atom_cross_att_decoder = _STATE["stock"]["enc"], _STATE["stock"]["dec"]
        DH.sample, DH.DiffusionHead = _STATE["stock"]["sample"], _STATE["stock"]["head"]
        _STATE["installed"] = False


def report() -> dict:
    return {"installed": _STATE["installed"], "precomputed": _STATE["precomputed"], "enc": _STATE["enc"], "dec": _STATE["dec"], "passed": _STATE["passed"],
            "aside": dict(_STATE["aside"])}
