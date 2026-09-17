# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import torch
import torch.nn as nn

from xfold import feat_batch, features
from xfold import of3
from xfold.nn import featurization
from xfold.nn.pairformer import EvoformerBlock, PairformerBlock
from xfold.nn.head import DistogramHead, ConfidenceHead
from xfold.nn.template import TemplateEmbedding
from xfold.nn import atom_cross_attention
from xfold.nn import diffusion_head


class Evoformer(nn.Module):
    def __init__(self, msa_channel: int = 64):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.msa_stack_num_layer = 4
        self.pairformer_num_layer = 48
        self.num_msa = 1024

        self.seq_channel = 384
        self.pair_channel = 128
        self.c_target_feat = 447

        self.left_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)
        self.right_single = nn.Linear(
            self.c_target_feat, self.pair_channel, bias=False)

        self.prev_embedding_layer_norm = nn.LayerNorm(self.pair_channel)
        self.prev_embedding = nn.Linear(
            self.pair_channel, self.pair_channel, bias=False)

        self.c_rel_feat = 139
        self.position_activations = nn.Linear(
            self.c_rel_feat, self.pair_channel, bias=False)

        self.bond_embedding = nn.Linear(
            1, self.pair_channel, bias=False)

        self.template_embedding = TemplateEmbedding(
            pair_channel=self.pair_channel)

        self.msa_activations = nn.Linear(34, self.msa_channel, bias=False)
        self.extra_msa_target_feat = nn.Linear(
            self.c_target_feat, self.msa_channel, bias=False)
        self.msa_stack = nn.ModuleList(
            [EvoformerBlock() for _ in range(self.msa_stack_num_layer)])

        self.single_activations = nn.Linear(
            self.c_target_feat, self.seq_channel, bias=False)

        self.prev_single_embedding_layer_norm = nn.LayerNorm(self.seq_channel)
        self.prev_single_embedding = nn.Linear(
            self.seq_channel, self.seq_channel, bias=False)

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True) for _ in range(self.pairformer_num_layer)])

    def _relative_encoding(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        max_relative_idx = 32
        max_relative_chain = 2

        rel_feat = featurization.create_relative_encoding(
            batch.token_features,
            max_relative_idx,
            max_relative_chain,
        ).to(dtype=pair_activations.dtype)

        pair_activations += self.position_activations(rel_feat)
        return pair_activations

    def _seq_pair_embedding(
        self, token_features: features.TokenFeatures, target_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated Pair embedding from sequence."""
        left_single = self.left_single(target_feat)[:, None]
        right_single = self.right_single(target_feat)[None]
        pair_activations = left_single + right_single

        mask = token_features.mask

        pair_mask = (mask[:, None] * mask[None, :]).to(dtype=left_single.dtype)

        return pair_activations, pair_mask

    def _embed_bonds(
        self, batch: feat_batch.Batch, pair_activations: torch.Tensor
    ) -> torch.Tensor:
        """Embeds bond features and merges into pair activations."""
        # Construct contact matrix.
        num_tokens = batch.token_features.token_index.shape[0]
        contact_matrix = torch.zeros(
            (num_tokens, num_tokens), dtype=pair_activations.dtype, device=pair_activations.device)

        tokens_to_polymer_ligand_bonds = (
            batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
        )
        gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
        gather_mask_polymer_ligand = (
            tokens_to_polymer_ligand_bonds.gather_mask.prod(dim=1).to(
                dtype=gather_idxs_polymer_ligand.dtype)[:, None]
        )
        # If valid mask then it will be all 1's, so idxs should be unchanged.
        gather_idxs_polymer_ligand = (
            gather_idxs_polymer_ligand * gather_mask_polymer_ligand
        )

        tokens_to_ligand_ligand_bonds = (
            batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
        )
        gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
        gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
            dim=1
        ).to(dtype=gather_idxs_ligand_ligand.dtype)[:, None]
        gather_idxs_ligand_ligand = (
            gather_idxs_ligand_ligand * gather_mask_ligand_ligand
        )

        gather_idxs = torch.concatenate(
            [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
        )
        contact_matrix[
            gather_idxs[:, 0], gather_idxs[:, 1]
        ] = 1.0
        if of3.OF3:
            # OF3 weights were trained with a symmetric bond matrix
            contact_matrix[gather_idxs[:, 1], gather_idxs[:, 0]] = 1.0

        # Because all the padded index's are 0's.
        contact_matrix[0, 0] = 0.0

        bonds_act = self.bond_embedding(contact_matrix[:, :, None])

        return pair_activations + bonds_act

    def _embed_template_pair(
        self,
        batch: feat_batch.Batch,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embeds Templates and merges into pair activations."""
        templates = batch.templates
        asym_id = batch.token_features.asym_id

        dtype = pair_activations.dtype
        multichain_mask = (asym_id[:, None] ==
                           asym_id[None, :]).to(dtype=dtype)

        template_act = self.template_embedding(
            query_embedding=pair_activations,
            templates=templates,
            multichain_mask_2d=multichain_mask,
            padding_mask_2d=pair_mask
        )

        return pair_activations + template_act

    def _embed_process_msa(
        self, msa_batch: features.MSA,
        pair_activations: torch.Tensor,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor
    ) -> torch.Tensor:
        """Processes MSA and returns updated pair activations."""
        dtype = pair_activations.dtype

        msa_batch = featurization.shuffle_msa(msa_batch)
        msa_batch = featurization.truncate_msa_batch(msa_batch, self.num_msa)

        msa_mask = msa_batch.mask.to(dtype=dtype)
        msa_feat = featurization.create_msa_feat(msa_batch).to(dtype=dtype)

        msa_activations = self.msa_activations(msa_feat)
        msa_activations += self.extra_msa_target_feat(target_feat)[None]

        # Evoformer MSA stack.
        for msa_block in self.msa_stack:
            msa_activations, pair_activations = msa_block(
                msa=msa_activations,
                pair=pair_activations,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
            )

        return pair_activations

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )

        pair_activations += self.prev_embedding(
            self.prev_embedding_layer_norm(prev['pair']))

        pair_activations = self._relative_encoding(batch, pair_activations)

        pair_activations = self._embed_bonds(
            batch=batch, pair_activations=pair_activations
        )

        pair_activations = self._embed_template_pair(
            batch=batch,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
        )

        pair_activations = self._embed_process_msa(
            msa_batch=batch.msa,
            pair_activations=pair_activations,
            pair_mask=pair_mask,
            target_feat=target_feat,
        )

        single_activations = self.single_activations(target_feat)
        single_activations += self.prev_single_embedding(
            self.prev_single_embedding_layer_norm(prev['single']))

        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, batch.token_features.mask)

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output


class AlphaFold3(nn.Module):
    def __init__(self, num_recycles: int = 10, num_samples: int = 5, diffusion_steps: int = 200):
        super(AlphaFold3, self).__init__()

        self.num_recycles = num_recycles
        self.num_samples = num_samples
        self.diffusion_steps = diffusion_steps

        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer_conditioning = atom_cross_attention.AtomCrossAttEncoder()

        self.evoformer = Evoformer()

        self.diffusion_head = diffusion_head.DiffusionHead()

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()

    def create_target_feat_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        target_feat = featurization.create_target_feat(
            batch,
            append_per_atom_features=False,
        )

        enc = self.evoformer_conditioning(
            token_atoms_act=None,
            trunk_single_cond=None,
            trunk_pair_cond=None,
            batch=batch,
        )

        target_feat = torch.concatenate([target_feat, enc.token_act], dim=-1)

        return target_feat

    def _apply_denoising_step(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,
        noise_level_prev: torch.Tensor,
        mask: torch.Tensor,
        noise_level: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        positions = diffusion_head.random_augmentation(
            positions=positions, mask=mask
        )

        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        noise = noise_scale * \
            torch.randn(size=positions.shape, device=noise_scale.device)
        positions_noisy = positions + noise

        if getattr(self.diffusion_head, "use_step_graph", False):
            positions_denoised = self.diffusion_head.forward_graphed(positions_noisy, t_hat, batch, embeddings, True)
        else:
            positions_denoised = self.diffusion_head(positions_noisy=positions_noisy,
                                                     noise_level=t_hat,
                                                     batch=batch,
                                                     embeddings=embeddings,
                                                     use_conditioning=True)
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out, noise_level

    # ------------------------------------------------------------------ lever 'sbatch': the SAMPLE-BATCHED sampler
    # Stock (below) runs `for sample: for step`: S x T serial single-sample denoiser calls, every one re-reading the step-invariant statics for
    # ONE sample's arithmetic. With `self.sample_batch = b >= 1` (build_model, lever 'sbatch') the loop is `for step: for chunk of <= b samples`:
    # each denoiser call advances a chunk of samples at ONE noise level — T x ceil(S / b) calls; the hoisted statics (pair conditioning, the 24
    # blocks' pair logits / DTK's bf16 pair bias, the atom encoder / decoder statics) and the per-step single conditioning are computed once and
    # SHARED by the chunk (never materialised x S: only activations carry the sample axis — DiffusionHead.forward is rank-polymorphic, the DTK
    # transformer reads the conditioning rows periodically); the whole batched step is ONE CUDA graph under 'stepgraph' (forward_graphed keys on
    # the chunk's shape). RNG: every draw stock makes — the initial positions (one draw for all samples), then per (sample, step) the rotation's
    # randn(2,3), the translation's randn(3) and the step noise — is made by the SAME call at the SAME position of the CUDA Philox stream
    # (diffusion_head.DrawPlan seeks the generator per unit; a counter-based generator returns the same bits wherever the call falls in program
    # order), so sample s of the batched sampler receives draw for draw the noise stock's serial loop gives sample s, and the generator is left
    # where stock leaves it. Numerics: the augmentation, the noise and the update are stock's per-sample statements (bitwise); the denoiser at
    # batch b is the tolerance class (batched GEMM / attention reduction order; b = 1 chunks reproduce the serial kernels' shapes).
    def _draw_rows(self, batch: feat_batch.Batch, n: int, num_samples: int) -> int:
        """The token rows the sampler's random draws are made at: the model's own length (stock). A package lever that draws at another length
        (canonical_noise: the input's real token count, zero rows laid up to ``n``) rebinds this on the instance."""
        return n

    def _apply_denoising_step_batched(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        positions: torch.Tensor,          # [b, N, A, 3]: samples s0 .. s0+b-1 at one noise level
        noise_level_prev: torch.Tensor,   # 0-dim (shared by the chunk: every sample of a step is at the same level)
        mask: torch.Tensor,
        noise_level: torch.Tensor,
        plan: "diffusion_head.DrawPlan",
        step_idx: int,
        s0: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma_0 * (noise_level > self.gamma_min)
        t_hat = noise_level_prev * (1 + gamma)

        noise_scale = self.noise_scale * \
            torch.sqrt(t_hat**2 - noise_level_prev**2)
        n = int(positions.shape[1])
        if getattr(self, "batched_prologue", False):             # lever 'prologue': the same draws per sample, the arithmetic once over the sample axis
            positions_noisy = diffusion_head.augment_and_noise_batched(positions, mask, noise_scale, plan, step_idx, s0, n)
        else:
            positions_noisy = torch.empty_like(positions)
            for j in range(int(positions.shape[0])):                 # per sample, stock's own statements on stock's own draws (bitwise the serial step's)
                plan.seek(s0 + j, step_idx)                          # the generator where stock's loop stands before this (sample, step)'s three draws
                augmented = diffusion_head.random_augmentation(
                    positions=positions[j], mask=mask
                )
                noise = noise_scale * diffusion_head.lay_rows(
                    torch.randn(size=(plan.rows,) + tuple(positions.shape[2:]), device=noise_scale.device), n, 0)
                positions_noisy[j] = augmented + noise

        if getattr(self.diffusion_head, "use_step_graph", False):
            positions_denoised = self.diffusion_head.forward_graphed(positions_noisy, t_hat, batch, embeddings, True)
        else:
            positions_denoised = self.diffusion_head(positions_noisy=positions_noisy,
                                                     noise_level=t_hat,
                                                     batch=batch,
                                                     embeddings=embeddings,
                                                     use_conditioning=True)
        grad = (positions_noisy - positions_denoised) / t_hat

        d_t = noise_level - t_hat
        positions_out = positions_noisy + self.step_scale * d_t * grad

        return positions_out, noise_level

    def _sample_diffusion_batched(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """``_sample_diffusion`` with the samples advanced together (lever 'sbatch', ``self.sample_batch`` = the most samples per denoiser call)."""
        mask = batch.predicted_structure_info.atom_mask
        num_samples = self.num_samples
        sub = max(1, min(int(self.sample_batch), num_samples))

        device = mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        n = int(mask.shape[0])
        rows = int(self._draw_rows(batch, n, num_samples))       # stock: n; canonical_noise: the input's real token count
        plan = diffusion_head.DrawPlan(device, num_samples, self.diffusion_steps, rows, tuple(mask.shape[1:]), torch.float32)
        positions = diffusion_head.lay_rows(plan.initial(), n, 1).contiguous()   # stock's ONE initial draw for all samples, at `rows` token rows
        positions *= noise_levels[0]

        noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

        if getattr(self.diffusion_head, "use_hoist", False) or getattr(self.diffusion_head, "use_step_graph", False):
            self.diffusion_head.prime_static(batch, embeddings)   # step-invariant conditioning, once per trajectory (shared by every sample)

        counts = getattr(self, "_sbatch_counts", None)
        for step_idx in range(self.diffusion_steps):
            for s0 in range(0, num_samples, sub):
                s1 = min(num_samples, s0 + sub)
                positions[s0:s1], noise_level[s0:s1] = self._apply_denoising_step_batched(
                    batch, embeddings, positions[s0:s1], noise_level[s0], mask, noise_levels[1 + step_idx], plan, step_idx, s0)
                if counts is not None:
                    counts["batched_calls" if s1 - s0 > 1 else "single_calls"] += 1
        plan.finish()                                             # the generator where stock's serial loop leaves it
        if counts is not None:
            counts["trajectories"] += int(num_samples); counts["chunk"] = sub; counts["rng"] = plan.kind
            counts["prologue"] = "batched" if getattr(self, "batched_prologue", False) else "per_sample"

        final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))

        return {'atom_positions': positions, 'mask': final_dense_atom_mask}

    def _sample_diffusion(
        self,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample using denoiser on batch."""

        if getattr(self, "sample_batch", 0):
            return self._sample_diffusion_batched(batch, embeddings)   # lever 'sbatch': the samples advanced together per step (above)

        mask = batch.predicted_structure_info.atom_mask
        num_samples = self.num_samples

        device = mask.device

        noise_levels = diffusion_head.noise_schedule(
            torch.linspace(0, 1, self.diffusion_steps + 1, device=device))

        positions = torch.randn(
            (num_samples,) + mask.shape + (3,), device=device)
        positions *= noise_levels[0]

        noise_level = torch.tile(noise_levels[None, 0], (num_samples,))

        if getattr(self.diffusion_head, "use_hoist", False) or getattr(self.diffusion_head, "use_step_graph", False):
            self.diffusion_head.prime_static(batch, embeddings)   # step-invariant conditioning, once per trajectory

        for sample_idx in range(num_samples):
            for step_idx in range(self.diffusion_steps):
                positions[sample_idx], noise_level[sample_idx] = self._apply_denoising_step(
                    batch, embeddings, positions[sample_idx], noise_level[sample_idx], mask, noise_levels[1 + step_idx])

        final_dense_atom_mask = torch.tile(mask[None], (num_samples, 1, 1))

        return {'atom_positions': positions, 'mask': final_dense_atom_mask}

    @staticmethod
    def prep_batch(batch: dict[str, torch.Tensor]) -> feat_batch.Batch:
        """dict of feature tensors -> Batch, applying the OF3 atomic-number shift (0-indexed elements) when of3.OF3."""
        if isinstance(batch, feat_batch.Batch):
            return batch
        if of3.OF3:
            batch = dict(batch)
            batch['ref_element'] = torch.clamp(batch['ref_element'] - 1, min=0)
        return feat_batch.Batch.from_data_dict(batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self.prep_batch(batch)
        num_res = batch.num_res

        target_feat = self.create_target_feat_embedding(batch)

        embeddings = {
            'pair': torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            ),
            'target_feat': target_feat,  # type: ignore
        }

        # AF3 runs num_recycles + 1 Evoformer passes (hk.fori_loop(0, num_recycles + 1)); upstream xfold ran one pass too few.
        for _ in range(self.num_recycles + 1):
            embeddings = self.evoformer(
                batch=batch,
                prev=embeddings,
                target_feat=target_feat
            )

        samples = self._sample_diffusion(batch, embeddings)

        confidence_output_per_sample = []
        for sample_dense_atom_position in samples['atom_positions']:
            confidence_output_per_sample.append(self.confidence_head(
                dense_atom_positions=sample_dense_atom_position,
                embeddings=embeddings,
                seq_mask=batch.token_features.mask,
                token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
                asym_id=batch.token_features.asym_id
            ))

        confidence_output = {}
        for key in confidence_output_per_sample[0]:
            confidence_output[key] = torch.stack(
                [sample[key] for sample in confidence_output_per_sample], dim=0)

        distogram = self.distogram_head(batch, embeddings)

        return {
            'diffusion_samples': samples,
            'distogram': distogram,
            **confidence_output,
        }
