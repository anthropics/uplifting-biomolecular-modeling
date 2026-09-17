# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import dataclasses
from typing import Optional

import torch

from xfold import of3
import torch.nn as nn
import torch.nn.functional as F

from xfold import feat_batch
from xfold.nn import atom_layout, utils
from xfold.nn.diffusion_transformer import DiffusionCrossAttTransformer

from xfold import fastnn


@dataclasses.dataclass(frozen=True)
class AtomCrossAttEncoderOutput:
    token_act: torch.Tensor  # (num_tokens, ch)
    skip_connection: torch.Tensor  # (num_subsets, num_queries, ch)
    queries_mask: torch.Tensor  # (num_subsets, num_queries)
    queries_single_cond: torch.Tensor  # (num_subsets, num_queries, ch)
    keys_mask: torch.Tensor  # (num_subsets, num_keys)
    keys_single_cond: torch.Tensor  # (num_subsets, num_keys, ch)
    pair_cond: torch.Tensor  # (num_subsets, num_queries, num_keys, ch)


class AtomCrossAttEncoder(nn.Module):
    TOKEN_AGG = None                # lever 'token_agg': the kit's af3t_token_agg module (af3_torch_api.build_model binds it when the lever is on) -- the atom -> token
    TOKEN_AGG_COUNTS = {"calls": 0, "refused": {}}   # aggregation below as ONE kernel; census (forward.json items: token_agg): calls served / refused by name (stock statements ran)
    def __init__(self,
                 per_token_channels: int = 384,
                 per_atom_channels: int = 128,
                 per_atom_pair_channels: int = 16,
                 with_token_atoms_act: bool = False,
                 with_trunk_single_cond: bool = False,
                 with_trunk_pair_cond: bool = False) -> None:
        super(AtomCrossAttEncoder, self).__init__()

        self.with_token_atoms_act = with_token_atoms_act
        self.with_trunk_single_cond = with_trunk_single_cond
        self.with_trunk_pair_cond = with_trunk_pair_cond

        self.c_positions = 3
        self.c_mask = 1
        self.c_element = 128
        self.c_charge = 1
        self.c_atom_name = 256
        self.c_pair_distance = 1
        self.per_token_channels = per_token_channels
        self.per_atom_channels = per_atom_channels
        self.per_atom_pair_channels = per_atom_pair_channels

        self.embed_ref_pos = nn.Linear(
            self.c_positions, self.per_atom_channels, bias=False)

        self.embed_ref_mask = nn.Linear(
            self.c_mask, self.per_atom_channels, bias=False)

        self.embed_ref_element = nn.Linear(
            self.c_element, self.per_atom_channels, bias=False)
        self.embed_ref_charge = nn.Linear(
            self.c_charge, self.per_atom_channels, bias=False)

        self.embed_ref_atom_name = nn.Linear(
            self.c_atom_name, self.per_atom_channels, bias=False)

        self.single_to_pair_cond_row = nn.Linear(
            self.per_atom_channels, self.per_atom_pair_channels, bias=False)
        self.single_to_pair_cond_col = nn.Linear(
            self.per_atom_channels, self.per_atom_pair_channels, bias=False)

        self.embed_pair_offsets = nn.Linear(
            self.c_positions, self.per_atom_pair_channels, bias=False)
        self.embed_pair_distances = nn.Linear(
            self.c_pair_distance, self.per_atom_pair_channels, bias=False)

        self.single_to_pair_cond_row_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False)

        self.single_to_pair_cond_col_1 = nn.Linear(
            128, self.per_atom_pair_channels, bias=False)

        self.embed_pair_offsets_1 = nn.Linear(
            self.c_positions, self.per_atom_pair_channels, bias=False)

        self.embed_pair_distances_1 = nn.Linear(
            1, self.per_atom_pair_channels, bias=False)

        self.embed_pair_offsets_valid = nn.Linear(
            1, self.per_atom_pair_channels, bias=False)

        self.pair_mlp_1 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)
        self.pair_mlp_2 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)
        self.pair_mlp_3 = nn.Linear(
            self.per_atom_pair_channels, self.per_atom_pair_channels, bias=False)

        self.c_query = 128
        self.atom_transformer_encoder = DiffusionCrossAttTransformer(
            c_query=self.c_query)

        self.project_atom_features_for_aggr = nn.Linear(
            self.c_query, self.per_token_channels, bias=False)
        
        if self.with_trunk_single_cond is True:
            self.c_trunk_single_cond = 384
            self.lnorm_trunk_single_cond = fastnn.LayerNorm(
                self.c_trunk_single_cond, bias=False)
            self.embed_trunk_single_cond = nn.Linear(
                self.c_trunk_single_cond, self.per_atom_channels, bias=False)

        if self.with_token_atoms_act is True:
            self.atom_positions_to_features = nn.Linear(
                self.c_positions, self.per_atom_channels, bias=False)
            
        if self.with_trunk_pair_cond is True:
            self.c_trunk_pair_cond = 128
            self.lnorm_trunk_pair_cond = fastnn.LayerNorm(
                self.c_trunk_pair_cond, bias=False)
            self.embed_trunk_pair_cond = nn.Linear(
                self.c_trunk_pair_cond, self.per_atom_pair_channels, bias=False)

    def _per_atom_conditioning(self, batch: feat_batch.Batch) -> tuple[torch.Tensor, torch.Tensor]:

        # Compute per-atom single conditioning
        # Shape (num_tokens, num_dense, channels)
        act = self.embed_ref_pos(batch.ref_structure.positions)
        act += self.embed_ref_mask(batch.ref_structure.mask[:, :, None].to(
            dtype=self.embed_ref_mask.weight.dtype))

        # Element is encoded as atomic number if the periodic table, so
        # 128 should be fine.
        act += self.embed_ref_element(F.one_hot(batch.ref_structure.element.to(
            dtype=torch.int64), 128).to(dtype=self.embed_ref_element.weight.dtype))
        act += self.embed_ref_charge(torch.arcsinh(
            batch.ref_structure.charge)[:, :, None])

        # Characters are encoded as ASCII code minus 32, so we need 64 classes,
        # to encode all standard ASCII characters between 32 and 96.
        atom_name_chars_1hot = F.one_hot(batch.ref_structure.atom_name_chars.to(
            dtype=torch.int64), 64).to(dtype=self.embed_ref_atom_name.weight.dtype)
        num_token, num_dense, _ = act.shape
        act += self.embed_ref_atom_name(
            atom_name_chars_1hot.reshape(num_token, num_dense, -1))

        act *= batch.ref_structure.mask[:, :, None]

        # Compute pair conditioning
        # shape (num_tokens, num_dense, num_dense, channels)
        # Embed single features
        row_act = self.single_to_pair_cond_row(torch.relu(act))
        col_act = self.single_to_pair_cond_col(torch.relu(act))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        # Embed pairwise offsets
        pair_act += self.embed_pair_offsets(batch.ref_structure.positions[:, :, None, :]
                                            - batch.ref_structure.positions[:, None, :, :])

        sq_dists = torch.sum(
            torch.square(
                batch.ref_structure.positions[:, :, None, :]
                - batch.ref_structure.positions[:, None, :, :]
            ),
            dim=-1,
        )
        pair_act += self.embed_pair_distances(1.0 /
                                              (1 + sq_dists[:, :, :, None]))

        return act, pair_act

    def compute_static(self, batch: feat_batch.Batch, trunk_single_cond=None, trunk_pair_cond=None) -> dict:
        """Everything in the encoder that does NOT depend on token_atoms_act (atom positions): per-atom single/pair conditioning,
        masks, trunk single/pair conditioning gathered to atoms, offsets, pair MLP, and the atom transformer's pair logits.
        In the diffusion sampler these are identical at every step -> DiffusionHead.prime_static hoists them (same kernels, run once)."""
        token_atoms_single_cond, _ = self._per_atom_conditioning(batch)
        token_atoms_mask = batch.predicted_structure_info.atom_mask

        queries_single_cond = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_single_cond,
            layout_axes=(-3, -2),
        )

        queries_mask = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atoms_mask,
            layout_axes=(-2, -1),
        )

        # If provided, broadcast single conditioning from trunk to all queries
        if trunk_single_cond is not None:
            trunk_single_cond = self.embed_trunk_single_cond(
                self.lnorm_trunk_single_cond(trunk_single_cond))
            queries_single_cond += atom_layout.convert(
                batch.atom_cross_att.tokens_to_queries,
                trunk_single_cond,
                layout_axes=(-2,),
            )

        # (sokrypton fork, unconditional) mask the query single conditioning
        queries_single_cond = queries_single_cond * queries_mask[..., None]

        keys_single_cond = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )
        keys_mask = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys, queries_mask, layout_axes=(
                -2, -1)
        )

        # Embed single features into the pair conditioning.
        # shape (num_subsets, num_queries, num_keys, ch)
        row_act = self.single_to_pair_cond_row_1(
            torch.relu(queries_single_cond))
        pair_cond_keys_input = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_single_cond,
            layout_axes=(-3, -2),
        )

        col_act = self.single_to_pair_cond_col_1(
            torch.relu(pair_cond_keys_input))
        pair_act = row_act[:, :, None, :] + col_act[:, None, :, :]

        if trunk_pair_cond is not None:
            trunk_pair_cond = self.embed_trunk_pair_cond(
                self.lnorm_trunk_pair_cond(trunk_pair_cond))
            
            # Create the GatherInfo into a flattened trunk_pair_cond from the
            # queries and keys gather infos.
            num_tokens = trunk_pair_cond.shape[0]
            # (num_subsets, num_queries)
            tokens_to_queries = batch.atom_cross_att.tokens_to_queries
            # (num_subsets, num_keys)
            tokens_to_keys = batch.atom_cross_att.tokens_to_keys
            # (num_subsets, num_queries, num_keys)
            trunk_pair_to_atom_pair = atom_layout.GatherInfo(
                gather_idxs=(
                    num_tokens * tokens_to_queries.gather_idxs[:, :, None]
                    + tokens_to_keys.gather_idxs[:, None, :]
                ),
                gather_mask=(
                    tokens_to_queries.gather_mask[:, :, None]
                    & tokens_to_keys.gather_mask[:, None, :]
                ),
                input_shape=torch.tensor((num_tokens, num_tokens), device=torch.device('cpu')),
            )
            # Gather the conditioning and add it to the atom-pair activations.
            pair_act += atom_layout.convert(
                trunk_pair_to_atom_pair, trunk_pair_cond, layout_axes=(-3, -2)
            )

        # Embed pairwise offsets
        queries_ref_pos = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.positions,
            layout_axes=(-3, -2),
        )
        queries_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            batch.ref_structure.ref_space_uid,
            layout_axes=(-2, -1),
        )
        keys_ref_pos = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_pos,
            layout_axes=(-3, -2),
        )
        keys_ref_space_uid = atom_layout.convert(
            batch.atom_cross_att.queries_to_keys,
            queries_ref_space_uid,  # (sokrypton fork, unconditional) keys gathered from the QUERIES layout
            layout_axes=(-2, -1),
        )

        offsets_valid = (
            queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
        )
        if of3.OF3:
            # OF3 excludes padded key atoms (ref_space_uid zero-fill collides with token 0)
            offsets_valid = offsets_valid & keys_mask[:, None, :].to(dtype=torch.bool)
        offsets = queries_ref_pos[:, :, None, :] - keys_ref_pos[:, None, :, :]

        pair_act += (self.embed_pair_offsets_1(offsets)
                     * offsets_valid[:, :, :, None])

        # Embed pairwise inverse squared distances
        sq_dists = torch.sum(torch.square(offsets), dim=-1)
        pair_act += self.embed_pair_distances_1(
            1.0 / (1 + sq_dists[:, :, :, None])) * offsets_valid[:, :, :, None]
        # Embed offsets valid mask
        pair_act += self.embed_pair_offsets_valid(offsets_valid[:, :, :, None].to(
            dtype=self.embed_pair_offsets_valid.weight.dtype))

        # Run a small MLP on the pair acitvations
        pair_act2 = self.pair_mlp_1(torch.relu(pair_act))
        pair_act2 = self.pair_mlp_2(torch.relu(pair_act2))
        pair_act += self.pair_mlp_3(torch.relu(pair_act2))

        enc_pair_logits = self.atom_transformer_encoder.compute_pair_logits(pair_act)
        out = dict(token_atoms_mask=token_atoms_mask, queries_single_cond=queries_single_cond, queries_mask=queries_mask,
                   keys_single_cond=keys_single_cond, keys_mask=keys_mask, pair_cond=pair_act, enc_pair_logits=enc_pair_logits)
        if AtomCrossAttEncoder.TOKEN_AGG is not None:              # lever 'token_agg': the aggregation kernel's step-invariant operands (slot weights = token atom mask x
            gi = batch.atom_cross_att.queries_to_token_atoms         # gather mask, 1 / clamp(atoms per token, eps), the gather rows), hoisted with the rest
            out.update(AtomCrossAttEncoder.TOKEN_AGG.operands(gi.gather_idxs, gi.gather_mask, token_atoms_mask))
        return out

    def _token_aggregate(self, queries_act: torch.Tensor, static: dict):
        """Lever 'token_agg': the two stock statements below (atom_layout.convert to the token-atoms layout + the masked mean of relu over each token's
        atoms) as ONE kernel over the projected atom activations [.., Q, 32, ch] -> token_act [.., N, ch] fp32 (af3t_token_agg: the gathered
        [.., N, A, ch] intermediate is never formed; fp32 accumulation over a token's atom slots in slot order -- the tolerance class). None when
        the kernel refuses this call by name (counted; the stock statements serve)."""
        K = AtomCrossAttEncoder.TOKEN_AGG; cnt = AtomCrossAttEncoder.TOKEN_AGG_COUNTS
        w, inv, idx = static["agg_w"], static["agg_inv"], static["agg_idx"]
        ch = int(queries_act.shape[-1])
        x = queries_act.reshape((-1, int(queries_act.shape[-3]) * int(queries_act.shape[-2]), ch))      # [S, Q*32, ch] (a view: the projection's output is contiguous)
        why = None
        if not x.is_cuda:
            why = "device:cpu"
        elif x.stride(-1) != 1:
            why = "layout"
        elif idx.dim() != 2 or w.shape != idx.shape or inv.shape[0] != idx.shape[0]:
            why = "operands"
        if why is None:
            try:
                token_act = K.token_mean_relu(x, idx, w, inv)                               # [S, N, ch] fp32
            except Exception as e:                                                       # noqa: BLE001 -- a launch the kernel cannot serve: named, the stock statements serve
                if torch.cuda.is_current_stream_capturing() or "out of memory" in str(e).lower():
                    raise                                                                # under a whole-step capture the error is the capture's; an out-of-memory propagates
                why = "kernel_error:%s" % type(e).__name__
        if why is not None:
            cnt["refused"][why] = cnt["refused"].get(why, 0) + 1
            return None
        cnt["calls"] += 1
        return token_act[0] if queries_act.dim() == 3 else token_act.view(tuple(queries_act.shape[:-3]) + tuple(token_act.shape[-2:]))

    def forward(
        self,
        batch: feat_batch.Batch,
        token_atoms_act: Optional[torch.Tensor] = None,
        trunk_single_cond: Optional[torch.Tensor] = None,
        trunk_pair_cond: Optional[torch.Tensor] = None,
        static: Optional[dict] = None,
    ) -> AtomCrossAttEncoderOutput:

        assert (token_atoms_act is not None) == self.with_token_atoms_act
        assert (trunk_single_cond is not None) == self.with_trunk_single_cond
        assert (trunk_pair_cond is not None) == self.with_trunk_pair_cond

        if static is None:
            static = self.compute_static(batch, trunk_single_cond, trunk_pair_cond)
        token_atoms_mask = static["token_atoms_mask"]; queries_single_cond = static["queries_single_cond"]; queries_mask = static["queries_mask"]
        keys_single_cond = static["keys_single_cond"]; keys_mask = static["keys_mask"]; pair_act = static["pair_cond"]

        if token_atoms_act is None:
            queries_act = queries_single_cond.clone()
        else:
            # Convert token_atoms_act to queries layout and map to per_atom_channels
            # (num_subsets, num_queries, channels)
            queries_act = atom_layout.convert(
                batch.atom_cross_att.token_atoms_to_queries,
                token_atoms_act,
                layout_axes=(-3, -2),
            )

            queries_act = self.atom_positions_to_features(queries_act)
            queries_act *= queries_mask[..., None]
            queries_act += queries_single_cond

        queries_act = self.atom_transformer_encoder(
            queries_act=queries_act,
            queries_mask=queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=keys_mask,
            queries_single_cond=queries_single_cond,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
            pair_logits=static.get("enc_pair_logits"),
            window=static.get("enc_window"),                  # lever 'atom_window': the encoder blocks' hoisted window operands (DiffusionHead.prime_static), else None
        )

        queries_act *= queries_mask[..., None]
        skip_connection = queries_act.clone()

        queries_act = self.project_atom_features_for_aggr(queries_act)

        token_act = self._token_aggregate(queries_act, static) if (AtomCrossAttEncoder.TOKEN_AGG is not None and "agg_w" in static) else None
        if token_act is None:                                    # stock: gather to the token-atoms layout, then the masked mean of relu over each token's atoms
            token_atoms_act = atom_layout.convert(
                batch.atom_cross_att.queries_to_token_atoms,
                queries_act,
                layout_axes=(-3, -2),
            )

            token_act = utils.mask_mean(                             # a leading sample axis on token_atoms_act (lever 'sbatch': [S, N, A, ch]) is served by the
                token_atoms_mask.reshape((1,) * (token_atoms_act.dim() - 3) + tuple(token_atoms_mask.shape))[..., None], torch.relu(token_atoms_act), dim=-2   # mask viewed [1.., N, A, 1] (a view:
            )                                                        # same values, same statement at S absent)

        return AtomCrossAttEncoderOutput(
            token_act=token_act,
            skip_connection=skip_connection,
            queries_mask=queries_mask,
            queries_single_cond=queries_single_cond,
            keys_mask=keys_mask,
            keys_single_cond=keys_single_cond,
            pair_cond=pair_act,
        )


class AtomCrossAttDecoder(nn.Module):
    def __init__(self) -> None:
        super(AtomCrossAttDecoder, self).__init__()

        self.per_atom_channels = 128

        self.project_token_features_for_broadcast = nn.Linear(
            768, self.per_atom_channels, bias=False)

        self.atom_transformer_decoder = DiffusionCrossAttTransformer(
            c_query=self.per_atom_channels)

        self.atom_features_layer_norm = fastnn.LayerNorm(
            self.per_atom_channels, bias=False)
        self.atom_features_to_position_update = nn.Linear(
            self.per_atom_channels, 3, bias=False)

    def forward(self,
                batch: feat_batch.Batch,
                token_act: torch.Tensor,  # (num_tokens, ch)
                enc: AtomCrossAttEncoderOutput,
                pair_logits: Optional[torch.Tensor] = None,    # HOIST: decoder atom-transformer pair logits
                window: Optional[dict] = None) -> torch.Tensor:   # lever 'atom_window': the decoder blocks' hoisted window operands
        token_act = self.project_token_features_for_broadcast(token_act)
        num_token, max_atoms_per_token = (
            batch.atom_cross_att.queries_to_token_atoms.shape
        )
        token_atom_act = torch.broadcast_to(                     # token_act [N, ch] (stock) or [S, N, ch] (lever 'sbatch'): the same view either way
            token_act[..., None, :],
            tuple(token_act.shape[:-1]) + (max_atoms_per_token, self.per_atom_channels),
        )
        queries_act = atom_layout.convert(
            batch.atom_cross_att.token_atoms_to_queries,
            token_atom_act,
            layout_axes=(-3, -2),
        )
        queries_act += enc.skip_connection
        queries_act *= enc.queries_mask[..., None]

        # Run the atom cross attention transformer.
        queries_act = self.atom_transformer_decoder(
            queries_act=queries_act,
            queries_mask=enc.queries_mask,
            queries_to_keys=batch.atom_cross_att.queries_to_keys,
            keys_mask=enc.keys_mask,
            queries_single_cond=enc.queries_single_cond,
            keys_single_cond=enc.keys_single_cond,
            pair_cond=enc.pair_cond,
            pair_logits=pair_logits,
            window=window,
        )

        queries_act *= enc.queries_mask[..., None]
        queries_act = self.atom_features_layer_norm(queries_act)
        queries_position_update = self.atom_features_to_position_update(
            queries_act)
        position_update = atom_layout.convert(
            batch.atom_cross_att.queries_to_token_atoms,
            queries_position_update,
            layout_axes=(-3, -2),
        )
        return position_update
