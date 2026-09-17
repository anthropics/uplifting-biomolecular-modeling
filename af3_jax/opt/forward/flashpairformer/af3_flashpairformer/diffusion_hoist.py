"""diffusion_hoist.py - hoists the step-invariant diffusion conditioning out of AF3's 200-step sampling loop (JAX/Haiku, sokrypton
OF3-weights port `of3_weights=True` and the official layout). Stock recomputes, at every denoising step (and inside the 4x-unrolled hk.scan):
  (a) pair conditioning: LayerNorm+Linear over [N,N,c_pair+rel] and two pair transition blocks on [N,N,128] (f32), and
  (b) the per-block pair-logit projections of all 24 diffusion-transformer blocks (LayerNorm+Linear [N,N,128] -> 16 heads each).
Both depend only on the trunk embeddings/batch, not on the step. This patch computes them ONCE per sample call, in the same Haiku scopes
(same parameter names -> stock checkpoint unchanged), and feeds them into the loop. Arithmetic per element is unchanged (same ops, same
dtypes/precision); whether the result is bitwise-identical to stock depends on XLA compiling the hoisted ops identically (a property of the
compiled program, not guaranteed by this source).
install() / uninstall() rebind Model._sample_diffusion, DiffusionHead.__call__ and Transformer.__call__.
"""
import functools
import haiku as hk
import jax
import jax.numpy as jnp
from alphafold3.model import model as _model
from alphafold3.model.components import haiku_modules as hm, utils
from alphafold3.model.network import diffusion_head as DH, diffusion_transformer as DT, featurization

LS = "__layer_stack_no_per_layer"
_STOCK = {}


def _pair_conditioning(self, batch, embeddings, use_conditioning):
    """pair half of DiffusionHead._conditioning, verbatim ops."""
    pair_embedding = use_conditioning * embeddings["pair"]
    rel_features = featurization.create_relative_encoding(seq_features=batch.token_features, max_relative_idx=32, max_relative_chain=2).astype(pair_embedding.dtype)
    features_2d = jnp.concatenate([pair_embedding, rel_features], axis=-1)
    pair_cond = hm.Linear(self.config.conditioning.pair_channel, precision="highest", name="pair_cond_initial_projection")(
        hm.LayerNorm(use_fast_variance=False, create_offset=False, name="pair_cond_initial_norm")(features_2d))
    for idx in range(2):
        pair_cond += DT.transition_block(pair_cond, 2, self.global_config, name=f"pair_transition_{idx}")
    return pair_cond


def _single_conditioning(self, batch, embeddings, noise_level, use_conditioning):
    """single half of DiffusionHead._conditioning, verbatim ops (of3 and official branches)."""
    from alphafold3.constants import residue_names
    from alphafold3.model.network import noise_level_embeddings
    single_embedding = use_conditioning * embeddings["single"]
    target_feat = embeddings["target_feat"]
    features_1d = jnp.concatenate([single_embedding, target_feat], axis=-1)
    if getattr(self.global_config, "of3_weights", False):
        num_af3_restypes = residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP
        pad = jnp.zeros_like(features_1d[..., :1]); single_channels = single_embedding.shape[-1]
        aatype_end = single_channels + num_af3_restypes; profile_end = aatype_end + num_af3_restypes
        features_1d = jnp.concatenate([features_1d[..., :aatype_end], pad, features_1d[..., aatype_end:profile_end], pad, features_1d[..., profile_end:]], axis=-1)
    single_cond = hm.LayerNorm(use_fast_variance=False, create_offset=False, name="single_cond_initial_norm")(features_1d)
    single_cond = hm.Linear(self.config.conditioning.seq_channel, precision="highest", name="single_cond_initial_projection")(single_cond)
    if getattr(self.global_config, "of3_weights", False):
        _dim = len(noise_level_embeddings._WEIGHT)
        fw = hk.get_parameter("fourier_embedding_weight", shape=[_dim], dtype=jnp.float32, init=hk.initializers.Constant(0.0))
        fb = hk.get_parameter("fourier_embedding_bias", shape=[_dim], dtype=jnp.float32, init=hk.initializers.Constant(0.0))
        noise_embedding = noise_level_embeddings.noise_embeddings(sigma_scaled_noise_level=noise_level / DH.SIGMA_DATA, weight=fw, bias=fb)
    else:
        noise_embedding = noise_level_embeddings.noise_embeddings(sigma_scaled_noise_level=noise_level / DH.SIGMA_DATA)
    single_cond += hm.Linear(self.config.conditioning.seq_channel, precision="highest", name="noise_embedding_initial_projection")(
        hm.LayerNorm(use_fast_variance=False, create_offset=False, name="noise_embedding_initial_norm")(noise_embedding))
    for idx in range(2):
        single_cond += DT.transition_block(single_cond, 2, self.global_config, name=f"single_transition_{idx}")
    return single_cond


class HoistTransformer(DT.Transformer):
  """Transformer with two extra modes (of3_weights layout: per-block LN+projection):
  mode='precompute' -> per-block pair logits [n_super, super_size, H, N, N] (f32); mode='step' -> blocks using the precomputed logits."""

  def __call__(self, act, mask, single_cond, pair_cond, mode="stock", pair_logits_pre=None):
    of3 = getattr(self.global_config, "of3_weights", False)
    if mode == "precompute" and not of3:
        return None                      # official layout: single shared pair LayerNorm + per-super-block projection stay in the step (not hoisted here)
    if mode == "stock" or not of3 or (mode == "step" and pair_logits_pre is None):
        return super().__call__(act, mask, single_cond, pair_cond)
    nsb = self.config.num_blocks // self.config.super_block_size; sbs = self.config.super_block_size
    if mode == "precompute":
        def blk(c):
            pair_act = hm.LayerNorm(name="pair_input_layer_norm", use_fast_variance=False, create_offset=False)(pair_cond)
            lg = hm.Linear(self.config.attention.num_head, name="pair_logits_projection")(pair_act)
            return c, jnp.transpose(lg, [2, 0, 1])
        def sb(c):
            return hk.experimental.layer_stack(sbs, with_per_layer_inputs=True, name=LS)(blk)(c)
        _, logits = hk.experimental.layer_stack(nsb, with_per_layer_inputs=True, name=LS)(sb)(jnp.zeros((), jnp.float32))
        return logits
    assert mode == "step"
    def block(act, block_pair_logits):
        act += DT.self_attention(act, mask, block_pair_logits, self.config.attention, self.global_config, single_cond, name=self.name)
        act += DT.transition_block(act, self.config.num_intermediate_factor, self.global_config, single_cond, name=self.name)
        return act, None
    def super_block(act, lgs):
        act, _ = hk.experimental.layer_stack(sbs, with_per_layer_inputs=True, name=LS)(block)(act, lgs)
        return act, None
    act, _ = hk.experimental.layer_stack(nsb, with_per_layer_inputs=True, name=LS)(super_block)(act, pair_logits_pre)
    return act


class HoistDiffusionHead(DH.DiffusionHead):
  """DiffusionHead whose __call__ has mode='precompute' (step-invariant pair conditioning + per-block pair logits, once) and mode='step'."""

  def __call__(self, positions_noisy, noise_level, batch, embeddings, use_conditioning, mode="stock", pre=None):
    if mode == "stock":
        return super().__call__(positions_noisy, noise_level, batch, embeddings, use_conditioning)
    from alphafold3.model.network import atom_cross_attention
    with utils.bfloat16_context():
        if mode == "precompute":
            pair_cond = _pair_conditioning(self, batch, embeddings, use_conditioning)
            pair_cond32 = jnp.asarray(pair_cond, dtype=jnp.float32)
            logits = HoistTransformer(self.config.transformer, self.global_config)(None, None, None, pair_cond32, mode="precompute")
            return dict(pair_cond=pair_cond, pair_logits=logits)
        assert mode == "step" and pre is not None
        trunk_single_cond = _single_conditioning(self, batch, embeddings, noise_level, use_conditioning)
        trunk_pair_cond = pre["pair_cond"]
        sequence_mask = batch.token_features.mask; atom_mask = batch.predicted_structure_info.atom_mask
        act = positions_noisy * atom_mask[..., None]
        act = act / jnp.sqrt(noise_level**2 + DH.SIGMA_DATA**2)
        enc = atom_cross_attention.atom_cross_att_encoder(token_atoms_act=act, trunk_single_cond=embeddings["single"], trunk_pair_cond=trunk_pair_cond,
                                                           config=self.config, global_config=self.global_config, batch=batch, name="diffusion")
        act = enc.token_act
        act = jnp.asarray(act, dtype=jnp.float32)
        act += hm.Linear(act.shape[-1], precision="highest", initializer=self.global_config.final_init, name="single_cond_embedding_projection")(
            hm.LayerNorm(use_fast_variance=False, create_offset=False, name="single_cond_embedding_norm")(trunk_single_cond))
        act = jnp.asarray(act, dtype=jnp.float32)
        trunk_single_cond = jnp.asarray(trunk_single_cond, dtype=jnp.float32); trunk_pair_cond = jnp.asarray(trunk_pair_cond, dtype=jnp.float32)
        sequence_mask = jnp.asarray(sequence_mask, dtype=jnp.float32)
        transformer = HoistTransformer(self.config.transformer, self.global_config)
        act = transformer(act, sequence_mask, trunk_single_cond, trunk_pair_cond, mode="step", pair_logits_pre=pre["pair_logits"])
        act = hm.LayerNorm(use_fast_variance=False, create_offset=False, name="output_norm")(act)
        position_update = atom_cross_attention.atom_cross_att_decoder(token_act=act, enc=enc, config=self.config, global_config=self.global_config, batch=batch, name="diffusion")
        skip_scaling = DH.SIGMA_DATA**2 / (noise_level**2 + DH.SIGMA_DATA**2)
        out_scaling = noise_level * DH.SIGMA_DATA / jnp.sqrt(noise_level**2 + DH.SIGMA_DATA**2)
    return (skip_scaling * positions_noisy + out_scaling * position_update) * atom_mask[..., None]


def sample_diffusion_hoisted(self, batch, embeddings, *, sample_config):
    # rng: draw the sampler key FIRST, at the same position in Haiku's rng sequence as stock (hk.experimental.layer_stack consumes an rng
    # draw in apply, so precomputing before this draw would hand the sampler a different noise key - a seed change, not an arithmetic one).
    key = hk.next_rng_key()
    pre = self.diffusion_module(None, None, batch=batch, embeddings=embeddings, use_conditioning=True, mode="precompute")   # once per sample call, outside hk.scan/hk.vmap
    denoising_step = functools.partial(self.diffusion_module, batch=batch, embeddings=embeddings, use_conditioning=True, mode="step", pre=pre)
    return DH.sample(denoising_step=denoising_step, batch=batch, key=key, config=sample_config)


def install():
    """Model.__init__ constructs diffusion_head.DiffusionHead(...) at trace time -> rebinding the module attribute is sufficient; the transformer
    subclass is referenced directly by HoistDiffusionHead. Default Transformer name ('transformer') and DiffusionHead name are inherited."""
    if not _STOCK:
        _STOCK.update(sample=_model.Model._sample_diffusion, dh_cls=DH.DiffusionHead)
    _model.Model._sample_diffusion = hk.transparent(sample_diffusion_hoisted)
    DH.DiffusionHead = HoistDiffusionHead
    return "hoist"


def uninstall():
    if _STOCK:
        _model.Model._sample_diffusion = _STOCK["sample"]; DH.DiffusionHead = _STOCK["dh_cls"]
    return "stock"
