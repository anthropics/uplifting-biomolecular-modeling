import itertools

import equinox as eqx
import jax
from jaxtyping import Float, Array, Int

from collections.abc import Callable
import jax.numpy as jnp
import numpy as np
import gemmi

from ..common import LossTerm

# 64 bins, 0.0–32.0 Å, shared across all model backends
PAE_BINS = np.arange(start=0.25, stop=32.0, step=0.5)


def reduce_samples(values, auxiliary, reduction, num_samples):
    """Reduce per-sample loss ``values`` and reorder scalar aux metrics.

    ``values`` has shape ``(num_samples,)``. Per-sample scalar aux leaves (also
    shape ``(num_samples,)``) are sorted into ascending-loss order and returned
    as lists; non-scalar leaves (predicted structures, full PSSMs, ...) pass
    through unchanged. Shared by every multi-sample structure-prediction loss.

    The sorted list is returned even for ``num_samples == 1`` (a 1-element
    list), so aux shape is uniform across sample counts and models. Consumers
    that look up a metric by key (e.g. ``biohub_optimizer``) match the key
    anywhere in the tree path so the list wrapping does not hide it.
    """
    sortperm = jnp.argsort(values)

    def _sort_if_scalar(value):
        if isinstance(value, jax.Array) and value.shape == (num_samples,):
            return list(value[sortperm])
        return value

    return reduction(values), jax.tree.map(_sort_if_scalar, auxiliary)


class StructureModelOutput(eqx.Module):
    distogram_logits: Float[Array, "N N Bins"]
    distogram_bins: Float[Array, " Bins"]
    plddt: Float[Array, " N"]
    pae: Float[Array, "N N"]
    pae_logits: Float[Array, "N N Bins"]
    pae_bins: Float[Array, " Bins"]
    structure_coordinates: Array  # all-atom coords, shape varies by model
    backbone_coordinates: Float[Array, "N 4 3"]
    # Per-token identity, a 20-d distribution in mosaic-20 / `common.TOKENS`
    # order ("ARNDCQEGHILKMFPSTWYV"). INVARIANT: every model wrapper emits this
    # order (af2 / boltz / boltz2 / protenix / of3 / esmfold2 all do — their
    # native alphabets coincide here or are remapped to it), so `argmax` indexes
    # the canonical alphabet. Relied on by `to_pdb`, proteina, protein_mpnn.
    full_sequence: Float[Array, "N 20"]
    asym_id: Float[Array, " N"]
    residue_idx: Int[Array, " N"]
    # Canonical atom37 view of the predicted heavy atoms (see
    # `mosaic.losses.atom37.ATOM37_NAMES`, == AlphaFold `atom_types` order).
    # Populated by every model wrapper.
    atom37_coords: Float[Array, "N 37 3"]
    atom37_mask: Float[Array, "N 37"]

    def to_structure(self) -> gemmi.Structure:
        """Predicted structure as a gemmi.Structure from the atom37
        view.

        Identity comes from `full_sequence` (mosaic-20 order, see field note),
        chains from `asym_id`, atoms + names from `atom37_coords` / `atom37_mask`
        (canonical AF2 `atom_types` order), b-factors from `plddt` (scaled to the
        0–100 pLDDT convention), and residue numbers directly from the model's `residue_idx`, without
        renumbering.

        Covers protein complexes only: atom37 is heavy-atom / standard-residue,
        so ligand or non-canonical chains aren't represented (use a model's
        native chain_infos writer for those).
        """
        from mosaic.alphafold.common import residue_constants
        from mosaic.alphafold.common.protein import PDB_CHAIN_IDS

        mask = np.asarray(self.atom37_mask)
        aatype = np.asarray(self.full_sequence).argmax(-1).astype(np.int32)
        asym = np.asarray(self.asym_id).astype(np.int32)
        resnum = np.asarray(self.residue_idx).astype(np.int32)
        coords = np.asarray(self.atom37_coords)
        confidence = np.asarray(self.plddt) * 100.0
        restypes = residue_constants.restypes + ["X"]

        structure = gemmi.Structure()
        model = gemmi.Model("1")
        for chain_id in dict.fromkeys(asym.tolist()):
            chain = gemmi.Chain(PDB_CHAIN_IDS[chain_id])
            for i in np.flatnonzero(asym == chain_id):
                one_letter = restypes[aatype[i]]
                residue = gemmi.Residue()
                residue.name = residue_constants.restype_1to3.get(one_letter, "UNK")
                residue.seqid = gemmi.SeqId(int(resnum[i]), " ")
                for atom_name, position, present in zip(
                    residue_constants.atom_types, coords[i], mask[i]
                ):
                    if present < 0.5:
                        continue
                    atom = gemmi.Atom()
                    atom.name = atom_name
                    atom.element = gemmi.Element(atom_name[0])
                    atom.pos = gemmi.Position(*map(float, position))
                    atom.occ = 1.0
                    atom.b_iso = float(confidence[i])
                    residue.add_atom(atom)
                chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        return structure

    def chain_pair_iptm(self) -> dict[tuple[int, int], float]:
        """Interface pTM (iPTM) for every unordered pair of chains.

        Same formula as `IPTMLoss` — `predicted_tm_score` over the inter-chain
        residue pairs, maxed over the alignment residue — but with the pair mask
        restricted to one chain pair at a time, so a multi-chain complex yields a
        value per interface instead of one lumped binder-vs-rest score. For a
        two-chain complex the single entry equals the overall iPTM.

        Not JIT-able (Python loop over a dynamic number of chains); call it on a
        concrete output, outside `jit`. Keys are `(asym_id, asym_id)` index pairs
        with the lower index first.
        """
        asym = np.asarray(self.asym_id).astype(np.int32)
        chains = sorted({int(a) for a in asym.tolist()})
        out: dict[tuple[int, int], float] = {}
        for a, b in itertools.combinations(chains, 2):
            inter = ((asym == a)[:, None] & (asym == b)[None, :]) | (
                (asym == b)[:, None] & (asym == a)[None, :]
            )
            iptm = predicted_tm_score(
                logits=self.pae_logits,
                bin_centers=self.pae_bins,
                pair_mask=jnp.asarray(inter),
            ).max()
            out[(a, b)] = float(iptm)
        return out


def interaction_prediction_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    asym_id: jnp.ndarray | None = None,
    pae_cutoff: float = 10.0,
) -> jnp.ndarray:
    probs = jax.nn.softmax(logits, axis=-1)
    pae = jnp.sum(probs * bin_centers, axis=-1)

    pair_mask = jnp.ones_like(pae, dtype=bool)
    pair_mask *= asym_id[:, None] != asym_id[None, :]

    # only include residue pairs below the pae_cutoff
    pair_mask *= pae < pae_cutoff
    n_residues = jnp.sum(pair_mask, axis=-1, keepdims=True)

    # Compute adjusted d_0(num_res) per residue  as defined by eqn. (15) in
    # Dunbrack, R., "What's wrong with AlphaFold’s ipTM score and how to fix it."
    # 2025: https://pmc.ncbi.nlm.nih.gov/articles/PMC11844409/
    d0 = 1.24 * (jnp.clip(n_residues, min=27) - 15) ** (1.0 / 3) - 1.8

    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    normed_residue_mask = pair_mask / (1e-8 + n_residues)
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment


def predicted_tm_score(
    logits: jnp.ndarray,
    bin_centers: jnp.ndarray,
    pair_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    num_res = logits.shape[0]
    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = max(num_res, 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

    # Convert logits to probs.
    probs = jax.nn.softmax(logits, axis=-1)

    # TM-Score term for every bin.
    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
    # E_distances tm(distance).
    predicted_tm_term = jnp.sum(probs * tm_per_bin, axis=-1)

    if pair_mask is None:
        pair_mask = jnp.ones(shape=(num_res, num_res), dtype=bool)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask
    normed_residue_mask = pair_residue_weights / (
        1e-8 + jnp.sum(pair_residue_weights, axis=-1, keepdims=True)
    )
    per_alignment = jnp.sum(predicted_tm_term * normed_residue_mask, axis=-1)
    return per_alignment


def contact_cross_entropy(
    distogram_logits: Float[Array, "N N Bins"],
    contact_dist: float,
    bins: Float[Array, " Bins"],
) -> Float[Array, "... N N"]:
    """Compute partial entropy (under distogram) that D_ij < contact_dist."""
    assert bins.shape[-1] == distogram_logits.shape[-1]
    assert distogram_logits.ndim == 3

    distogram_logits = jax.nn.log_softmax(distogram_logits)

    mask = bins < contact_dist

    px_ = jax.nn.softmax(distogram_logits, axis=-1, where=mask)

    return (px_ * distogram_logits).sum(-1)


def contact_log_probability(
    distogram_logits: Float[Array, "... N N 64"],
    contact_dist: float,
    bins: Float[Array, " Bins"],
) -> Float[Array, "... N N"]:
    """Compute log probability (under distogram) that D_ij < contact_dist."""
    assert bins.shape[-1] == distogram_logits.shape[-1]
    assert distogram_logits.ndim == 3
    distogram_logits = jax.nn.log_softmax(distogram_logits)
    mask = bins < contact_dist
    return jax.nn.logsumexp(distogram_logits, where=mask, axis=-1)


class WithinBinderContact(LossTerm):
    max_contact_distance: float = 14.0
    min_sequence_separation: int = 8
    num_contacts_per_residue: int = 25

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact_intra = contact_cross_entropy(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_contact_distance,
            bins=output.distogram_bins,
        )
        # only count binder-binder contacts with sequence sep > min_sequence_separation
        within_binder_mask = (
            jnp.abs(jnp.arange(binder_len)[:, None] - jnp.arange(binder_len)[None, :])
            > self.min_sequence_separation
        )
        # for each position in binder find positions most likely to make contact

        # JAX/XLO has a bizarre issue with top_k when used inside vmap _only_ when used on multiple GPUs.
        # so for now we sort instead of using top_k
        # binder_binder_max_p, _ = jax.vmap(
        #     lambda lcp: jax.lax.top_k(lcp, self.num_contacts_per_residue)
        # )(log_contact_intra + (1 - within_binder_mask) * -30)
        # average_log_prob = binder_binder_max_p.mean()

        sorted_log_probs = jnp.sort(
            log_contact_intra + (1 - within_binder_mask) * -30, descending=True, axis=-1
        )
        top_k_log_probs = sorted_log_probs[:, : self.num_contacts_per_residue]
        top_k_mean = top_k_log_probs.mean(axis=-1)

        average_log_prob = top_k_mean.mean()
        return -average_log_prob, {"intra_contact": average_log_prob}


class ESMFoldInterContact(LossTerm):
    """Subtly variation on `BinderTargetContact` used in ESMFold2 hallucination:
    for each _target_ residue use the most confident binder residue. Defaults to22"""

    contact_distance: float = 22.0
    num_contacts: int = 1

    def __call__(
        self, sequence: Float[Array, "N 20"], output: StructureModelOutput, key
    ):
        # if output.distogram_bins[-1] < self.contact_distance + 5.0:
        #     print(f"WARNING: ESMFoldInterContact using contact distance of {self.contact_distance: 0.2f} but output.distogram_bins[-1] = {output.distogram_bins[-1]: 0.2f}")
        binder_len = sequence.shape[0]
        block = output.distogram_logits[binder_len:, :binder_len]  # [Lt, Lb, B]
        con_loss = -contact_cross_entropy(
            block, self.contact_distance, output.distogram_bins
        )
        per_target = jnp.sort(con_loss, axis=-1)[:, : self.num_contacts].mean(-1)
        loss = per_target.mean()
        return loss, {"esmfold_inter_contact": loss}


class BinderTargetContact(LossTerm):
    paratope_idx: list[int] | None = None
    paratope_size: int | None = None
    contact_distance: float = 20.0
    epitope_idx: list[int] | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact_inter = contact_cross_entropy(
            output.distogram_logits[:binder_len, binder_len:],
            self.contact_distance,
            bins=output.distogram_bins,
        )
        if self.epitope_idx is not None:
            log_contact_inter = log_contact_inter[:, self.epitope_idx]

        # see above note about JAX/XLO issue with top_k inside vmap
        # binder_target_max_p = jax.vmap(lambda v: jax.lax.top_k(v, 3)[0])(
        #     log_contact_inter
        # ).mean(-1)
        sorted_log_probs = jnp.sort(log_contact_inter, descending=True, axis=-1)
        binder_target_max_p = sorted_log_probs[:, :3].mean(axis=-1)

        # log probability of contacting target for each position in binder
        if self.paratope_idx is not None:
            binder_target_max_p = binder_target_max_p[self.paratope_idx]
        if self.paratope_size is not None:
            binder_target_max_p = jax.lax.top_k(
                binder_target_max_p, self.paratope_size
            )[0]

        average_log_prob = binder_target_max_p.mean()
        return -average_log_prob, {"target_contact": average_log_prob}


class HelixLoss(LossTerm):
    max_distance: float = 6.0
    target_value: float = -2.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        log_contact = contact_log_probability(
            output.distogram_logits[:binder_len, :binder_len],
            self.max_distance,
            bins=output.distogram_bins,
        )
        value = jnp.diagonal(log_contact, 3).mean()

        loss = jax.nn.elu(self.target_value - value)

        return loss, {"helix": loss}


class ESMFoldGlobularity(LossTerm):
    target_radius: float | None = None
    clamp_max: float = 27.0

    def __call__(
        self, sequence: Float[Array, "N 20"], output: StructureModelOutput, key
    ):
        n = sequence.shape[0]
        probs = jax.nn.softmax(output.distogram_logits[:n, :n], axis=-1)  # [Lb, Lb, B]
        bins = jnp.minimum(output.distogram_bins, self.clamp_max)
        e_sq_dist = (probs * jnp.square(bins)).sum(-1)
        rg = jnp.sqrt(jnp.tril(e_sq_dist, k=-1).sum() / (n * n))
        rg_th = 2.38 * n**0.365 if self.target_radius is None else self.target_radius
        loss = jax.nn.elu(rg - rg_th)
        return loss, {"esmfold_globularity": loss, "esmfold_rg": rg}


class DistogramRadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # TODO: Why RMSE instead of MAE?
        binder_len = sequence.shape[0]
        dgram_radius_of_gyration = jnp.sqrt(
            jnp.fill_diagonal(
                (
                    jax.nn.softmax(output.distogram_logits)[:binder_len, :binder_len]
                    * (output.distogram_bins[None, None, :] ** 2)
                ).sum(-1),  # expected squared distance
                0,
                inplace=False,
            ).mean()
            + 1e-8
        )

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th), {
            "radius_of_gyration": dgram_radius_of_gyration
        }


class MAERadiusOfGyration(LossTerm):
    target_radius: float | None = None

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]

        dgram_radius_of_gyration = jnp.fill_diagonal(
            (
                jax.nn.softmax(output.distogram_logits)[:binder_len, :binder_len]
                * (output.distogram_bins[None, None, :])
            ).sum(-1),  # expected squared distance
            0,
            inplace=False,
        ).mean()

        rg_th = (
            2.38 * binder_len**0.365
            if self.target_radius is None
            else self.target_radius
        )
        return jax.nn.elu(dgram_radius_of_gyration - rg_th), {
            "radius_of_gyration": dgram_radius_of_gyration
        }


class DistogramCE(LossTerm):
    f: Float[Array, "... Bins"]
    name: str
    l: float = -np.inf
    u: float = np.inf

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        # expand dims so self.f is broadcastable to network_output["pdistogram"] of size (N, N, Bins)
        f = jnp.expand_dims(self.f, [i for i in range(3 - self.f.ndim)])

        ce = -jnp.fill_diagonal(
            (
                jax.nn.log_softmax(output.distogram_logits)[:binder_len, :binder_len]
                * f
            ).sum(-1),
            0,
            inplace=False,
        ).mean()

        return ce.clip(self.l, self.u), {self.name: ce}


class PLDDTLoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        plddt = output.plddt[:binder_len].mean()
        return -plddt, {"plddt": plddt}


class WithinBinderPAE(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        pae_within = jnp.fill_diagonal(
            output.pae[:binder_len, :binder_len], 0, inplace=False
        ).mean()
        return pae_within, {"bb_pae": pae_within}


class BinderTargetPAE(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        pae = output.pae[:binder_len, binder_len:].mean()
        return pae, {"bt_pae": pae}


class TargetBinderPAE(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        pae = output.pae[binder_len:, :binder_len].mean()
        return pae, {"tb_pae": pae}


class IPTMLoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # binder - target iptm -- we override asym-id in the case of multi-chain targets
        N = output.full_sequence.shape[0]
        asym_id = jnp.concatenate(
            (jnp.zeros(sequence.shape[0]), jnp.ones(N - sequence.shape[0]))
        ).astype(jnp.int32)
        pair_mask = asym_id[:, None] != asym_id[None, :]
        iptm = predicted_tm_score(
            logits=output.pae_logits,
            bin_centers=output.pae_bins,
            pair_mask=pair_mask,
        ).max()
        return -iptm, {"iptm": iptm}


class DistogramIPTMProxy(LossTerm):
    """Distogram iPTM proxy (Algorithm 15, ESM2 paper supplement A.3.3).

    Cheap differentiable surrogate for iPTM built only from the binder→target
    block of the predicted distogram (no diffusion sampling, no confidence
    head). For each pair (i, j) of binder→target tokens:

        p_full = softmax_b(D_ij)
        p_cut  = softmax_b(D_ij − ∞·(1 − m))    where m = [bin < cutoff]
        S_ij   = − Σ_b p_cut · log p_full
               = H(p_cut) − log P_contact
               (P_contact = Σ_b m · p_full = total mass in contact bins)

    `S_ij` is small when the model is both confident this pair contacts
    (high `P_contact`) and confident about the contact distance (low
    `H(p_cut)`). The proxy averages the `binder_len` lowest scores across
    all (i, j) pairs (the "minibinder" preset from the paper) and maps to
    [0, 1] via `clip(1 − S̄ / log n_contact_bins, 0, 1)`. Loss = −proxy.
    """

    contact_distance: float = 8.0

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        D_bt = output.distogram_logits[:binder_len, binder_len:]  # [N, L-N, B]
        bins = output.distogram_bins  # [B]

        m_b = bins < self.contact_distance  # [B] bool
        n_contact_bins = m_b.sum()

        log_p_full = jax.nn.log_softmax(D_bt, axis=-1)
        p_cut = jax.nn.softmax(D_bt, axis=-1, where=m_b)
        S = -(p_cut * log_p_full).sum(axis=-1)  # [N, L-N]

        # Mean of the k = binder_len smallest pair scores.
        S_flat = S.reshape(-1)
        bottom_k = -jax.lax.top_k(-S_flat, k=binder_len)[0]
        S_bar = bottom_k.mean()

        log_norm = jnp.log(n_contact_bins.astype(S_bar.dtype))
        proxy = jnp.clip(1.0 - S_bar / log_norm, 0.0, 1.0)
        return -proxy, {"distogram_iptm": proxy}


class BinderTargetIPTM(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        # binder - target iptm -- we override asym-id in the case of multi-chain targets
        N = output.full_sequence.shape[0]
        asym_id = jnp.concatenate(
            (jnp.zeros(sequence.shape[0]), jnp.ones(N - sequence.shape[0]))
        ).astype(jnp.int32)
        pair_mask = asym_id[:, None] != asym_id[None, :]
        bt_iptm = predicted_tm_score(
            logits=output.pae_logits,
            bin_centers=output.pae_bins,
            pair_mask=pair_mask,
        )[: sequence.shape[0]].max()  # limit to binder index
        return -bt_iptm, {"bt_iptm": bt_iptm}


class BinderPTMLoss(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        num_res = output.full_sequence.shape[0]
        pair_mask = jnp.zeros(shape=(num_res, num_res), dtype=bool)
        pair_mask = pair_mask.at[:binder_len, :binder_len].set(True)
        ptm = predicted_tm_score(
            logits=output.pae_logits,
            bin_centers=output.pae_bins,
            pair_mask=pair_mask,
        ).max()
        return -ptm, {"binder_ptm": ptm}


class BinderTargetIPSAE(LossTerm):
    reduce: Callable = jnp.max

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        N = output.full_sequence.shape[0]
        binder_len = sequence.shape[0]
        # override asym-id in the case of multi-chain targets
        asym_id = jnp.concatenate(
            (jnp.zeros(binder_len), jnp.ones(N - binder_len))
        ).astype(jnp.int32)
        bt_ipsae = self.reduce(
            interaction_prediction_score(
                asym_id=asym_id,
                logits=output.pae_logits,
                bin_centers=output.pae_bins,
                pae_cutoff=10.0,
            )[:binder_len]
        )
        return -bt_ipsae, {"bt_ipsae": bt_ipsae}


class TargetBinderIPSAE(LossTerm):
    reduce: Callable = jnp.max

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        N = output.full_sequence.shape[0]
        binder_len = sequence.shape[0]
        # override asym-id in the case of multi-chain targets
        asym_id = jnp.concatenate(
            (jnp.zeros(binder_len), jnp.ones(N - binder_len))
        ).astype(jnp.int32)
        tb_ipsae = self.reduce(
            interaction_prediction_score(
                asym_id=asym_id,
                logits=output.pae_logits,
                bin_centers=output.pae_bins,
                pae_cutoff=10.0,
            )[binder_len:]
        )
        return -tb_ipsae, {"tb_ipsae": tb_ipsae}


class IPSAE_min(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        bt_ipsae = (
            -1
            * (
                BinderTargetIPSAE()(
                    sequence=sequence,
                    output=output,
                    key=key,
                )
            )[0]
        )
        tb_ipsae = (
            -1
            * (
                TargetBinderIPSAE()(
                    sequence=sequence,
                    output=output,
                    key=key,
                )
            )[0]
        )
        ipsae_min = jnp.minimum(bt_ipsae, tb_ipsae)

        return -ipsae_min, {"ipsae_min": ipsae_min}


class ActualRadiusOfGyration(LossTerm):
    target_radius: float

    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        binder_len = sequence.shape[0]
        first_atom_coords = output.backbone_coordinates[:binder_len, 0]
        rg = jnp.sqrt(
            ((first_atom_coords - first_atom_coords.mean(0)) ** 2).sum(-1).mean()
        )

        return jax.nn.elu(rg - self.target_radius), {"actual_rg": rg}


class pTMEnergy(LossTerm):
    def __call__(
        self,
        sequence: Float[Array, "N 20"],
        output: StructureModelOutput,
        key,
    ):
        len_binder = sequence.shape[0]
        logits = output.pae_logits
        num_res = logits.shape[0]
        # Clip num_res to avoid negative/undefined d0.
        clipped_num_res = max(num_res, 19)

        d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

        pae_bin_centers = output.pae_bins

        g_d_b = 1.0 / (1 + jnp.square(pae_bin_centers) / jnp.square(d0))
        energy = jax.scipy.special.logsumexp(a=logits, b=g_d_b, axis=-1)
        # return negative mean over cross-chain pairs
        binder_target = energy[:len_binder, len_binder:].mean()
        target_binder = energy[len_binder:, :len_binder].mean()
        E = -(binder_target + target_binder) / 2
        return E, {"pTMEnergy": E}
