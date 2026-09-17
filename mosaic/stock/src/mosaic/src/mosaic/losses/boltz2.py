from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import boltz.data.const as const
import boltz.main as boltz_main
import equinox as eqx
import gemmi
import jax
import joltz
import numpy as np
import torch
from boltz.model.models.boltz2 import Boltz2
from boltz.data.const import ref_atoms
from jax import numpy as jnp
from jaxtyping import Array, Float, PyTree
from joltz import TrunkState


from ..common import LinearCombination, LossTerm
from ..cache import resolve_cache
from .atom37 import ATOM37_INDEX, scatter_atom37
from .structure_prediction import PAE_BINS, StructureModelOutput, reduce_samples


def _build_boltz_atom37_table() -> np.ndarray:
    """Boltz `res_type` slot → atom37 lookup `[N_slots, max_tokatom]`.

    Slots 2..21 of `res_type` are the 20 standard amino acids in mosaic
    order (`ref_atoms[3letter]` gives per-residue atom names). Returns an
    int table where -1 means "atom not in atom37 layout for this residue".
    """
    n_slots = len(const.tokens)
    aa_3letter = const.tokens[2:22]  # 'ALA' .. 'VAL'
    max_tokatom = max(len(ref_atoms[r]) for r in aa_3letter)
    table = -np.ones((n_slots, max_tokatom), dtype=np.int32)
    for slot, three_letter in enumerate(const.tokens):
        if three_letter not in ref_atoms:
            continue
        for tokatom_idx, atom_name in enumerate(ref_atoms[three_letter]):
            if atom_name in ATOM37_INDEX:
                table[slot, tokatom_idx] = ATOM37_INDEX[atom_name]
    return table


_BOLTZ_TOKATOM_TO_ATOM37 = _build_boltz_atom37_table()


def load_boltz2(checkpoint_path: Path | None = None):
    checkpoint_path = resolve_cache(checkpoint_path, "boltz", "boltz2_conf.ckpt")
    if not checkpoint_path.exists():
        print(f"Downloading Boltz checkpoint to {checkpoint_path}")
        cache = checkpoint_path.parent
        cache.mkdir(parents=True, exist_ok=True)
        boltz_main.download_boltz2(cache)

    torch_model = Boltz2.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        map_location="cpu",
        # Note: these args ARE NOT USED during prediction, but are needed to load the model
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 25,
            "diffusion_samples": 1,
        },
        diffusion_process_args=asdict(boltz_main.Boltz2DiffusionParams()),
        # ema=False,
        msa_args=asdict(
            boltz_main.MSAModuleArgs(
                subsample_msa=True,
                num_subsampled_msa=1024,
                use_paired_feature=True,
            )
        ),
        pairformer_args=asdict(boltz_main.PairformerArgsV2()),
    ).eval()

    model = joltz.from_torch(torch_model)
    _model_params, _model_static = eqx.partition(model, eqx.is_inexact_array)
    return eqx.combine(jax.device_put(_model_params), _model_static)


## Duplicated code
class StructureWriter:
    writer: boltz_main.BoltzWriter
    atom_pad_mask: torch.Tensor
    record: any
    out_dir: str
    temp_dir_handle: TemporaryDirectory

    def __init__(
        self,
        *,
        features_dict,
        target_dir: Path,
        output_dir: Path,
        temp_dir_handle: TemporaryDirectory,
    ):
        self.writer = boltz_main.BoltzWriter(
            data_dir=target_dir,
            output_dir=output_dir,
            output_format="mmcif",
            boltz2=True,
        )
        self.atom_pad_mask = features_dict["atom_pad_mask"].unsqueeze(0)
        self.record = features_dict["record"][0]
        self.out_dir = output_dir
        self.temp_dir_handle = temp_dir_handle

    def __call__(self, sample_atom_coords) -> gemmi.Structure:
        confidence = torch.ones(1)

        pred_dict = {
            "exception": False,
            "coords": torch.tensor(np.array(sample_atom_coords)).unsqueeze(0),
            "masks": self.atom_pad_mask,
            "confidence_score": confidence,
        }
        self.writer.write_on_batch_end(
            None,
            None,
            pred_dict,
            None,
            {"record": [self.record]},
            None,
            None,
        )
        return gemmi.read_structure(
            str((Path(self.out_dir) / self.record.id) / f"{self.record.id}_model_0.cif")
        )


def load_features_and_structure_writer(
    input_yaml_str: str,
    cache=None,
) -> PyTree:
    print("Loading data")
    cache = resolve_cache(cache, "boltz", create=True)
    out_dir_handle = (
        TemporaryDirectory()
    )  # this is sketchy -- we have to remember not to let this get garbage collected
    out_dir = Path(out_dir_handle.name)
    # dump the yaml to a file
    input_data_path = out_dir / "input.yaml"
    input_data_path.write_text(input_yaml_str)
    data = boltz_main.check_inputs(input_data_path)
    # Process inputs
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"
    manifest = boltz_main.process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        boltz2=True,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    if manifest is None:
        print("Something odd happened with manifest, trying to reload.")
        manifest = boltz_main.Manifest.load(processed_dir / "manifest.json")

    processed = boltz_main.BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Create data module
    data_module = boltz_main.Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=0,
        mol_dir=mol_dir,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )

    # Load the features for the single example
    features_dict = list(data_module.predict_dataloader())[0]

    # convert features to numpy arrays
    features = {k: np.array(v) for k, v in features_dict.items() if k != "record"}

    ## one-hot the MSA

    features["msa"] = jax.nn.one_hot(features["msa"], const.num_tokens)
    # fix up some dtypes
    # features["method_feature"] = features["method_feature"].astype(np.int32)

    writer = StructureWriter(
        features_dict=features_dict,
        target_dir=processed.targets_dir,
        output_dir=out_dir / "output",
        temp_dir_handle=out_dir_handle,
    )

    return jax.tree.map(jnp.array, features), writer


def set_binder_sequence(
    new_sequence: Float[Array, "N 20"],
    features: PyTree,
):
    """Replace features related to first N tokens with `new_sequence.` Used for hallucination/binder design."""
    # features = jax.tree.map(lambda v: v.astype(jnp.float32), features)
    features["res_type"] = features["res_type"].astype(jnp.float32)
    features["msa"] = features["msa"].astype(jnp.float32)
    features["profile"] = features["profile"].astype(jnp.float32)
    assert len(new_sequence.shape) == 2
    assert new_sequence.shape[1] == 20
    binder_len = new_sequence.shape[0]

    # We only use the standard 20 amino acids, but boltz has 33 total tokens.
    # zero out non-standard AA types
    zero_padded_sequence = jnp.pad(new_sequence, ((0, 0), (2, 11)))
    n_msa = features["msa"].shape[1]
    print("n_msa", n_msa)

    # We assume there are no MSA hits for the binder sequence
    binder_profile = jnp.zeros_like(features["profile"][0, :binder_len])
    binder_profile = binder_profile.at[:binder_len].set(zero_padded_sequence) / n_msa
    binder_profile = binder_profile.at[:, 1].set((n_msa - 1) / n_msa)

    return features | {
        "res_type": features["res_type"]
        .at[0, :binder_len, :]
        .set(zero_padded_sequence),
        "msa": features["msa"].at[0, 0, :binder_len, :].set(zero_padded_sequence),
        "profile": features["profile"].at[0, :binder_len].set(binder_profile),
    }


BOLTZ2_DISTOGRAM_BINS = np.linspace(start=2.0, stop=22.0, num=64)


def boltz2_trunk(
    model: joltz.Joltz2,
    features: PyTree,
    *,
    recycling_steps: int,
    deterministic: bool,
    initial_recycling_state: TrunkState | None = None,
    key: jax.Array,
):
    """Run embedding + trunk recycling. Returns (initial_embedding, trunk_state)."""
    initial_embedding = model.embed_inputs(features)

    if initial_recycling_state is None:
        state = TrunkState(
            s=jnp.zeros_like(initial_embedding.s_init),
            z=jnp.zeros_like(initial_embedding.z_init),
        )
    else:
        state = initial_recycling_state

    def body_fn(carry, _):
        trunk_state, key = carry
        trunk_state = jax.tree.map(jax.lax.stop_gradient, trunk_state)
        trunk_state, key = model.trunk_iteration(
            trunk_state,
            initial_embedding,
            features,
            key=key,
            deterministic=deterministic,
        )
        return (trunk_state, key), None

    (final_state, _), _ = jax.lax.scan(
        body_fn,
        (state, key),
        None,
        length=recycling_steps,
    )
    return initial_embedding, final_state


def boltz2_forward_from_trunk(
    model: joltz.Joltz2,
    features: PyTree,
    initial_embedding,
    trunk_state: TrunkState,
    *,
    num_sampling_steps: int,
    deterministic: bool,
    key: jax.Array,
) -> StructureModelOutput:
    """Run distogram, structure, and confidence from pre-computed trunk state."""
    distogram_logits = model.distogram_module(trunk_state.z)[0, :, :, 0, :]

    q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
        model.diffusion_conditioning(
            trunk_state.s,
            trunk_state.z,
            initial_embedding.relative_position_encoding,
            features,
        )
    )
    with jax.default_matmul_precision("float32"):
        structure_coordinates = model.structure_module.sample(
            s_trunk=trunk_state.s,
            s_inputs=initial_embedding.s_inputs,
            feats=features,
            num_sampling_steps=num_sampling_steps,
            atom_mask=features["atom_pad_mask"],
            multiplicity=1,
            diffusion_conditioning={
                "q": q,
                "c": c,
                "to_keys": to_keys,
                "atom_enc_bias": atom_enc_bias,
                "atom_dec_bias": atom_dec_bias,
                "token_trans_bias": token_trans_bias,
            },
            key=jax.random.fold_in(key, 2),
        )

    confidence = model.confidence_module(
        s_inputs=initial_embedding.s_inputs,
        s=trunk_state.s,
        z=trunk_state.z,
        x_pred=structure_coordinates,
        feats=features,
        pred_distogram_logits=distogram_logits[None],
        key=jax.random.fold_in(key, 5),
        deterministic=deterministic,
    )

    # Backbone coordinates (N, CA, C, O)
    features_unbatched = jax.tree.map(lambda x: x[0], features)
    assert ref_atoms["UNK"][:4] == ["N", "CA", "C", "O"]
    first_atom_idx = jax.vmap(lambda atoms: jnp.nonzero(atoms, size=1)[0][0])(
        features_unbatched["atom_to_token"].T
    )
    all_atom_coords = structure_coordinates[0]
    backbone_coordinates = jnp.stack(
        [all_atom_coords[first_atom_idx + i] for i in range(4)], -2
    )

    # Atom37 view: scatter all-atom coords into the canonical heavy-atom layout.
    # `atom_pad_mask` is 0 on padding atoms; sentinel them to -1 so they're dropped.
    n_tokens = features_unbatched["res_type"].shape[0]
    res_slot = features_unbatched["res_type"].argmax(-1)             # [N_token]
    atom_to_token = features_unbatched["atom_to_token"].argmax(-1)   # [N_atom]
    tokatom_idx = jnp.arange(atom_to_token.shape[0]) - first_atom_idx[atom_to_token]
    atom37_idx = jnp.asarray(_BOLTZ_TOKATOM_TO_ATOM37)[res_slot[atom_to_token], tokatom_idx]
    atom37_idx = jnp.where(
        features_unbatched["atom_pad_mask"] > 0.5, atom37_idx, jnp.int32(-1),
    )
    atom37_coords, atom37_mask = scatter_atom37(
        all_atom_coords, atom_to_token, atom37_idx, n_tokens,
    )

    return StructureModelOutput(
        distogram_logits=distogram_logits,
        distogram_bins=BOLTZ2_DISTOGRAM_BINS,
        plddt=confidence.plddt[0],
        pae=confidence.pae[0],
        pae_logits=confidence.pae_logits[0],
        pae_bins=PAE_BINS,
        structure_coordinates=structure_coordinates,
        backbone_coordinates=backbone_coordinates,
        full_sequence=features["res_type"][0][:, 2:22],
        asym_id=features["asym_id"][0],
        residue_idx=features["residue_index"][0],
        atom37_coords=atom37_coords,
        atom37_mask=atom37_mask,
    )


class Boltz2Loss(LossTerm):
    joltz2: joltz.Joltz2
    features: PyTree
    loss: LossTerm | LinearCombination
    deterministic: bool = True
    recycling_steps: int = 0
    sampling_steps: int = 25
    name: str = "boltz2"
    initial_recycling_state: TrunkState | None = None

    def __call__(self, sequence: Float[Array, "N 20"], key=None):
        """Compute the loss for a given sequence."""
        features = set_binder_sequence(sequence, self.features)

        initial_embedding, trunk_state = boltz2_trunk(
            self.joltz2, features,
            recycling_steps=self.recycling_steps,
            deterministic=self.deterministic,
            initial_recycling_state=self.initial_recycling_state,
            key=key,
        )
        output = boltz2_forward_from_trunk(
            self.joltz2, features, initial_embedding, trunk_state,
            num_sampling_steps=self.sampling_steps,
            deterministic=self.deterministic,
            key=key,
        )

        v, aux = self.loss(sequence=sequence, output=output, key=key)
        return v, {self.name: aux}


class MultiSampleBoltz2Loss(LossTerm):
    joltz2: joltz.Joltz2
    features: PyTree
    loss: LossTerm | LinearCombination
    deterministic: bool = True
    recycling_steps: int = 0
    sampling_steps: int = 25
    num_samples: int = 4
    name: str = "boltz2multi"
    initial_recycling_state: TrunkState | None = None
    reduction: any = jnp.mean
    """
        Run the structure and confidence modules multiple times from the same trunk output.
        When `reduction` is jnp.mean this is equivalent to the expected loss over multiple samples *assuming a deterministic trunk*, but faster.
        This will consume quite a bit of memory -- if you'd like to sacrifice some speed for memory, replace the vmap below with a jax.lax.map.
    """

    def __call__(self, sequence: Float[Array, "N 20"], key=None):
        """Compute the loss for a given sequence."""
        features = set_binder_sequence(sequence, self.features)

        initial_embedding, trunk_state = boltz2_trunk(
            self.joltz2, features,
            recycling_steps=self.recycling_steps,
            deterministic=self.deterministic,
            initial_recycling_state=self.initial_recycling_state,
            key=key,
        )

        def apply_loss_to_single_sample(key):
            output = boltz2_forward_from_trunk(
                self.joltz2, features, initial_embedding, trunk_state,
                num_sampling_steps=self.sampling_steps,
                deterministic=self.deterministic,
                key=key,
            )
            return self.loss(sequence=sequence, output=output, key=key)

        vs, auxs = jax.vmap(apply_loss_to_single_sample)(
            jax.random.split(key, self.num_samples)
        )
        return reduce_samples(vs, auxs, self.reduction, self.num_samples)
