"""
Utils for sampling from sequence design models.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from atomworks.io.parser import parse as aw_parse
from atomworks.io.utils import non_rcsb
from atomworks.io.utils.io_utils import to_cif_string
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

import caliby.data.const as const
from caliby.checkpoint_utils import get_cfg_from_ckpt, load_from_checkpoint
from caliby.data.data import to
from caliby.data.datasets.atomworks_sd_dataset import sd_collator
from caliby.data.transform.preprocess import preprocess_transform
from caliby.data.transform.sd_featurizer import sd_featurizer
from caliby.eval.eval_utils.inference_dataloader import InferenceDataLoader
from caliby.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from caliby.model.seq_denoiser.sd_model import SeqDenoiser
from caliby.weights import resolve_ckpt_path

_VALID_POS_CONSTRAINT_COLUMNS = [
    "pdb_key",
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
    "symmetry_pos",
]


def _validate_pos_constraint_df(pos_constraint_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Validate that a pos_constraint_df has correctly specified columns and returns a properly formatted DataFrame."""
    if pos_constraint_df is None:
        return None

    if not set(pos_constraint_df.columns).issubset(_VALID_POS_CONSTRAINT_COLUMNS):
        raise ValueError(
            f"Invalid columns in pos_constraint_df. Expected subset of {_VALID_POS_CONSTRAINT_COLUMNS}. "
            f"Found: {pos_constraint_df.columns}"
        )

    # Set index to pdb key.
    pos_constraint_df = pos_constraint_df.set_index("pdb_key")

    # Set empty string to NaN for easier parsing.
    pos_constraint_df = pos_constraint_df.replace("", np.nan)

    return pos_constraint_df


def get_seq_des_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Load in a sequence design model.
    Example config:

    seq_des_cfg:
        # MPNN args
        model_name: "atom_mpnn"  # ["atom_mpnn"]
            atom_mpnn:
                # Atom MPNN args
                atom_mpnn_cfg: caliby/configs/seq_des/inference.yaml
                atom_mpnn_ckpt:
    """
    model_name = cfg.model_name
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    ckpt_path = resolve_ckpt_path(cfg.atom_mpnn.ckpt_name_or_path)
    lit_sd_model = load_from_checkpoint(LitSeqDenoiser, ckpt_path).eval()
    model_cfg, _ = get_cfg_from_ckpt(ckpt_path)
    data_cfg = hydra.utils.instantiate(model_cfg.data)
    sampling_cfg = OmegaConf.load(cfg.atom_mpnn.sampling_cfg)
    sampling_cfg = OmegaConf.merge(sampling_cfg, OmegaConf.to_container(cfg.atom_mpnn.overrides, resolve=True))
    seq_des_model["model"] = lit_sd_model.model
    seq_des_model["data_cfg"] = data_cfg
    seq_des_model["sampling_cfg"] = sampling_cfg

    return seq_des_model


def run_seq_des(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_paths: list[str],
    device: str,
    out_dir: str,
    pos_constraint_df: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux
    will be a dictionary with the following keys:
        - "out_pdb": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample
    """
    # Set up outputs.
    outputs = defaultdict(list)
    sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    pos_constraint_df = _validate_pos_constraint_df(pos_constraint_df)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")

    # Load PDBs with prefetching.
    loader = InferenceDataLoader(
        pdb_paths,
        data_cfg=data_cfg,
        device=device,
        batch_size=sampling_cfg.batch_size,
        num_workers=sampling_cfg.num_workers,
    )

    # Begin sampling.
    pbar = tqdm(
        total=len(pdb_paths),
        desc=f"Sampling {len(pdb_paths)} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
    )
    for batch in loader:
        B = len(batch["example_id"])

        # Initialize seq_cond and atom_cond masks.
        batch = initialize_sampling_masks(batch)

        # Parse fixed positions.
        batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=sampling_cfg.verbose)

        # Restrict aatype sampling at certain positions.
        sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
        sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
            batch, pos_constraint_df, verbose=sampling_cfg.verbose
        )

        # Parse symmetry positions.
        sampling_inputs["symmetry_pos"] = parse_symmetry_pos_info(
            batch, pos_constraint_df, verbose=sampling_cfg.verbose
        )

        # Run sampling.
        id_to_atom_arrays, id_to_aux = model.sample(batch, sampling_inputs=sampling_inputs)

        # Save outputs.
        for example_id, atom_arrays in id_to_atom_arrays.items():
            aux = id_to_aux[example_id]
            sample_stems = [f"{example_id}_sample{si}" for si in range(len(atom_arrays))]

            # Save output atom arrays to cif files.
            for si, sample_stem in enumerate(sample_stems):
                out_file = f"{sample_out_dir}/{sample_stem}.cif"
                atom_array = atom_arrays[si]
                with open(out_file, "w") as f:
                    f.write(to_cif_string(atom_array, include_nan_coords=False))

                outputs["example_id"].append(example_id)
                outputs["out_pdb"].append(out_file)
                outputs["U"].append(aux[si]["U"])
                outputs["input_seq"].append(aux[si]["input_seq"])

            # Get sampled sequences as a string, with ":" to separate chains.
            for si in range(len(atom_arrays)):
                chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_arrays[si])
                outputs["seq"].append(
                    ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                )

        pbar.update(B)
    pbar.close()

    return outputs


def run_seq_des_ensemble(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_to_conformers: dict[str, list[str]],  # maps from a given pdb name to its conformer pdb files
    device: str,
    out_dir: str,
    pos_constraint_df: pd.DataFrame | None = None,
    use_primary_res_type: bool = True,  # use res_type from primary structure. Otherwise use res_type from conformer pdb
) -> dict[str, Any]:
    """
    Given a list of processed structure files, run sequence design on them.
    """
    # Set up outputs.
    outputs = defaultdict(list)
    sample_out_dir = f"{out_dir}/samples"  # directory for output PDBs
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    pos_constraint_df = _validate_pos_constraint_df(pos_constraint_df)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")

    # Load PDBs with prefetching (each batch = all conformers for one PDB).
    loader = InferenceDataLoader.from_conformers(
        pdb_to_conformers,
        data_cfg=data_cfg,
        device=device,
        num_workers=sampling_cfg.num_workers,
    )

    # Begin sampling.
    for batch in tqdm(
        loader,
        total=len(loader),
        desc=f"Sampling {len(pdb_to_conformers)} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
    ):
        num_conformers = len(batch["example_id"])
        # Create tied_sampling_ids by tying all samples together.
        batch["tied_sampling_ids"] = torch.zeros(num_conformers, device=device, dtype=torch.long)

        # Use res_type from primary structure
        if use_primary_res_type:
            # Update restype in batch.
            batch["restype"] = batch["restype"][0:1].expand(
                num_conformers, *((batch["restype"].ndim - 1) * (-1,))
            )

            # Update atom array annotations.
            for i in range(1, len(batch["atom_array"])):
                atomwise_resnames = spread_token_wise(
                    batch["atom_array"][i],
                    const.AF3_ENCODING.idx_to_token[batch["restype"][0].argmax(dim=-1).cpu().numpy()],
                )
                batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

        # Ensure that all entries in the batch have the same residue and chain index so that they're aligned.
        if not sampling_cfg["ensemble_ignore_res_idx_mismatch"]:
            _validate_ensemble_alignment(batch)

        # Initialize seq_cond and atom_cond masks.
        batch = initialize_sampling_masks(batch)

        # Parse fixed positions.
        batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=sampling_cfg.verbose)

        # Restrict aatype sampling at certain positions.
        sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
        sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
            batch, pos_constraint_df, verbose=sampling_cfg.verbose
        )

        # Parse symmetry positions.
        sampling_inputs["symmetry_pos"] = parse_symmetry_pos_info(
            batch, pos_constraint_df, verbose=sampling_cfg.verbose
        )

        # Run sampling.
        id_to_atom_arrays, id_to_aux = model.sample(batch, sampling_inputs=sampling_inputs)

        # Save outputs.
        for example_id, atom_arrays in id_to_atom_arrays.items():
            aux = id_to_aux[example_id]
            sample_stems = [f"{example_id}_sample{si}" for si in range(len(atom_arrays))]

            # Save output atom arrays to cif files.
            for si in range(len(atom_arrays)):
                out_file = f"{sample_out_dir}/{sample_stems[si]}.cif"
                atom_array = atom_arrays[si]
                with open(out_file, "w") as f:
                    f.write(to_cif_string(atom_array, include_nan_coords=False))

                outputs["example_id"].append(example_id)
                outputs["out_pdb"].append(out_file)
                outputs["U"].append(aux[si]["U"])
                outputs["input_seq"].append(aux[si]["input_seq"])

            # Get sampled sequences as a string, with ":" to separate chains.
            for si in range(len(atom_arrays)):
                chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_arrays[si])
                outputs["seq"].append(
                    ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                )

    return outputs


def run_sidechain_packing(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_paths: list[str],
    device: str,
    out_dir: str,
) -> dict[str, Any]:
    """
    Given a list of PDBs, run sidechain packing on them.
    """
    # Set up outputs.
    outputs = defaultdict(list)
    sample_out_dir = f"{out_dir}/packed_samples"  # directory for output PDBs
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)

    # Load PDBs with prefetching.
    loader = InferenceDataLoader(
        pdb_paths,
        data_cfg=data_cfg,
        device=device,
        batch_size=sampling_cfg.batch_size,
        num_workers=sampling_cfg.num_workers,
    )

    # Begin sampling.
    sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
    pbar = tqdm(
        total=len(pdb_paths),
        desc=f"Packing sidechains for {len(pdb_paths)} PDBs...",
    )
    for batch in loader:
        B = len(batch["example_id"])

        # Initialize seq_cond and atom_cond masks.
        batch = initialize_sampling_masks(batch)

        # For sidechain packing, we default seq_cond_mask to be the full sequence.
        batch["seq_cond_mask"] = batch["token_resolved_mask"].clone()

        # Run sampling.
        atom_arrays = model.sidechain_pack(batch, sampling_inputs=sampling_inputs)

        # Save outputs.
        for example_id, atom_array in zip(batch["example_id"], atom_arrays):
            out_file = f"{sample_out_dir}/packed_{example_id}.cif"
            with open(out_file, "w") as f:
                f.write(to_cif_string(atom_array, include_nan_coords=False))
            outputs["example_id"].append(example_id)
            outputs["out_pdb"].append(out_file)

        pbar.update(B)
    pbar.close()

    return outputs


def score_samples(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_paths: list[str],
    device: str,
) -> dict[str, Any]:
    """
    Given a list of processed structure files, score the sequences on them.
    """
    # Set up outputs.
    outputs = defaultdict(list)

    # Load PDBs with prefetching.
    loader = InferenceDataLoader(
        pdb_paths,
        data_cfg=data_cfg,
        device=device,
        batch_size=sampling_cfg.batch_size,
        num_workers=sampling_cfg.num_workers,
    )
    pbar = tqdm(total=len(pdb_paths), desc=f"Scoring {len(pdb_paths)} PDBs...")

    # Begin scoring.
    for batch in loader:
        B = len(batch["example_id"])

        # Initialize seq_cond and atom_cond masks.
        batch = initialize_sampling_masks(batch)

        # Score samples.
        sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
        id_to_aux = model.score_samples(batch, sampling_inputs=sampling_inputs)

        # Store results.
        for example_id, aux in id_to_aux.items():
            outputs["example_id"].append(example_id)
            chain_info = non_rcsb.initialize_chain_info_from_atom_array(aux["atom_array"])
            outputs["seq"].append(
                ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
            )
            outputs["U"].append(aux["U"])
            outputs["U_i"].append(aux["U_i"])

        pbar.update(B)
    pbar.close()

    return outputs


def score_samples_ensemble(
    *,
    model: SeqDenoiser,
    data_cfg: DictConfig,
    sampling_cfg: DictConfig,
    pdb_to_conformers: dict[str, list[str]],  # maps from a given pdb name to its conformer pdb files
    device: str,
) -> dict[str, Any]:
    """
    Score sequences using Potts parameters computed from an ensemble of input backbones.
    """
    outputs = defaultdict(list)

    # Load PDBs with prefetching (each batch = all conformers for one PDB).
    loader = InferenceDataLoader.from_conformers(
        pdb_to_conformers,
        data_cfg=data_cfg,
        device=device,
        num_workers=sampling_cfg.num_workers,
    )
    for batch in tqdm(loader, total=len(loader), desc=f"Scoring {len(pdb_to_conformers)} PDBs..."):
        num_conformers = len(batch["example_id"])
        # Create tied_sampling_ids by tying all samples together.
        batch["tied_sampling_ids"] = torch.zeros(num_conformers, device=device, dtype=torch.long)

        # Ensure that all entries in the batch have the same residue and chain index so that they're aligned.
        if not sampling_cfg["ensemble_ignore_res_idx_mismatch"]:
            _validate_ensemble_alignment(batch)

        # Initialize seq_cond and atom_cond masks.
        batch = initialize_sampling_masks(batch)

        # Score samples.
        sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
        id_to_aux = model.score_samples(batch, sampling_inputs=sampling_inputs)

        # Store results.
        for example_id, aux in id_to_aux.items():
            outputs["example_id"].append(example_id)
            chain_info = non_rcsb.initialize_chain_info_from_atom_array(aux["atom_array"])
            outputs["seq"].append(
                ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
            )
            outputs["U"].append(aux["U"])
            outputs["U_i"].append(aux["U_i"])

    return outputs


def get_sd_batch(
    pdb_paths: list[str], *, data_cfg: DictConfig | None, device: str
) -> dict[str, Any]:
    """
    Given a list of pdb file paths, return a batch of sequence design model features.

    If data_cfg is None, use default cif parser args.
    """
    batch_examples = [get_sd_example(pdb_path, data_cfg) for pdb_path in pdb_paths]

    # Collate examples.
    batch = sd_collator(batch_examples)
    batch = to(batch, device)

    return batch


def get_sd_example(pdb_path: str, data_cfg: DictConfig | None) -> dict[str, Any]:
    """
    Given a pdb file path, return a dictionary of sequence design model features.

    If data_cfg is None, use default cif parser args.
    """
    # Preprocess the PDB file.
    example = preprocess_pdb(pdb_path, data_cfg)

    # Featurize the example.
    featurizer = sd_featurizer()
    example = featurizer(example)

    return example


def preprocess_pdb(pdb_path: str, data_cfg: DictConfig | None) -> dict[str, Any]:
    """
    Preprocess a PDB file using the preprocessing pipeline.
    """
    # Set up arguments for parsing cifs with AtomWorks.
    if data_cfg is None:
        default_cif_parser_args = {
            "add_missing_atoms": True,
            "remove_waters": True,
            "remove_ccds": [],
            "fix_ligands_at_symmetry_centers": True,
            "fix_arginines": True,
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
        }
        cif_parser_args = default_cif_parser_args
    else:
        cif_parser_args = OmegaConf.to_container(data_cfg.cif_parser_args, resolve=True)

    # Read in the CIF data.
    transformation_id = "1"  # Leep only the first assembly.
    cif_parser_args["build_assembly"] = [transformation_id]
    input_data = aw_parse(pdb_path, **cif_parser_args)
    atom_array_from_cif = input_data["assemblies"][transformation_id][0]  # (1, num_atoms) -> (num_atoms)

    # Run the preprocessing pipeline on the CIF data.
    pipeline = preprocess_transform()
    return pipeline(
        data={
            "example_id": Path(pdb_path).stem,
            "atom_array": atom_array_from_cif,
            "chain_info": input_data["chain_info"],
        }
    )


def initialize_sampling_masks(batch: dict[str, TensorType["b ..."]]) -> dict[str, torch.Tensor]:
    """
    Initialize the sampling masks for the batch. Modifies batch in place and returns it.
    """
    # Initialize sequence mask: always condition on non-protein or non-standard residues.
    standard_prot_mask = batch["is_protein"] & ~batch["is_atomized"]
    batch["seq_cond_mask"] = torch.zeros_like(batch["token_pad_mask"])
    batch["seq_cond_mask"] = torch.where(
        standard_prot_mask, torch.zeros_like(batch["seq_cond_mask"]), batch["token_resolved_mask"]
    )

    # Initialize atom mask: condition on backbone atoms, non-protein atoms, and non-standard residues.
    batch["atom_cond_mask"] = batch["prot_bb_atom_mask"]  # condition on backbone atoms

    # Condition on non-protein atoms and non-standard residues.
    atomwise_standard_prot_mask = (
        torch.gather(standard_prot_mask, dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]
    )
    batch["atom_cond_mask"] = torch.where(
        atomwise_standard_prot_mask.bool(), batch["atom_cond_mask"], batch["atom_resolved_mask"]
    )

    return batch


def parse_fixed_pos_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> dict[str, torch.Tensor]:
    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for seq-level and atom-level conditioning
    - possibly overridden "res_type"

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["fixed_pos_seq", "fixed_pos_scn"]
    where each entry is a comma-separated string of positions in the format "A1-100,B1-100", "A1-10,A15-20", or np.nan.
    """

    seq_cond_mask, atom_cond_mask = batch["seq_cond_mask"].clone(), batch["atom_cond_mask"].clone()

    if pos_constraint_df is None:
        if verbose:
            print("No fixed positions specified, redesigning all positions.")
        return batch

    for i, example_id in enumerate(batch["example_id"]):
        if verbose:
            print(f"\n======================== {example_id} ========================")

        if example_id not in pos_constraint_df.index:
            if verbose:
                print(f"No fixed positions found for {example_id}")
            continue

        ### Get fixed positions from df ###
        row = pos_constraint_df.loc[example_id]
        fixed_pos_seq, fixed_pos_scn = (
            row.get("fixed_pos_seq", np.nan),
            row.get("fixed_pos_scn", np.nan),
        )  # get fixed positions for this PDB

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{example_id}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, override_abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(
                fixed_pos_override_seq, example["atom_array"]
            )
            for abs_pos_i, aa in zip(override_abs_pos, override_aatypes):
                # Update restype in batch.
                batch["restype"][i, abs_pos_i] = F.one_hot(
                    torch.tensor(const.AF3_ENCODING.encode_aa_seq(aa), device=batch["restype"].device),
                    num_classes=const.AF3_ENCODING.n_tokens,
                )

            # Update atom array annotations.
            token_pad_mask = batch["token_pad_mask"][
                i
            ].bool()  # we need to get rid of padding since atom_arrays are not padded
            resnames = const.AF3_ENCODING.idx_to_token[batch["restype"][i][token_pad_mask].argmax(dim=-1).cpu().numpy()]
            atomwise_resnames = spread_token_wise(batch["atom_array"][i], resnames)
            batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{example_id}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, example["atom_array"])
            seq_cond_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                print("Fixed sequence:")
                visualize_conditioning_sequences(
                    example["atom_array"],
                    seq_cond_mask[i][example["token_pad_mask"].bool()],
                    example["asym_id"][example["token_pad_mask"].bool()],
                    example["feat_metadata"]["asym_name"],
                )
        else:
            if verbose:
                print(f"{example_id}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{example_id}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, example["atom_array"])
            scn_atom_mask = torch.isin(
                example["atom_to_token_map"],
                torch.tensor(abs_fixed_pos_scn, device=example["atom_to_token_map"].device),
            )
            atom_cond_mask[i] = torch.where(scn_atom_mask, example["atom_resolved_mask"], atom_cond_mask[i])

            # ensure that we're not fixing sidechains when we override the PDB sequence
            scn_cond_num_atoms = apply_token_wise(example["atom_array"], scn_atom_mask.cpu().numpy(), np.sum)
            if not pd.isna(fixed_pos_override_seq):
                assert (
                    scn_cond_num_atoms[override_abs_pos] == 0
                ).all(), "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            if verbose:
                print("Fixed sidechains:")
                visualize_conditioning_sequences(
                    example["atom_array"],
                    torch.tensor(scn_cond_num_atoms > 0),
                    example["asym_id"][example["token_pad_mask"].bool()].cpu(),
                    example["feat_metadata"]["asym_name"],
                )
        else:
            if verbose:
                print(f"{example_id}: No fixed sidechain positions specified.")

    # Update batch
    batch["seq_cond_mask"] = seq_cond_mask
    batch["atom_cond_mask"] = atom_cond_mask
    return batch


def parse_pos_restrict_aatype_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Given a pos_constraint_df containing position restrictions for each PDB, return:
    - a mask indicating which positions have restricted amino acid sampling
    - a mask indicating which amino acids are allowed at each position

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["pos_restrict_aatype"]
    where each entry is a comma-separated string of positions in the format "A1:AVG,B10:ILMV", or None.
    """
    B, N = batch["token_pad_mask"].shape
    K = const.AF3_ENCODING.n_tokens

    if pos_constraint_df is None:
        if verbose:
            print("No amino acid restrictions specified, allowing all amino acids at all positions.")
        return None

    # Initialize masks for the entire batch
    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["token_pad_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["token_pad_mask"].device)

    if verbose:
        print("\n************** Position-wise amino acid restrictions **************")

    for i, pdb_key in enumerate(batch["example_id"]):
        if pdb_key not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_key}: No amino acid restrictions specified.")
            continue

        # Get position restrictions from df
        row = pos_constraint_df.loc[pdb_key]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_key}: No position-wise amino acid restrictions specified.")
            continue

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        if verbose:
            print(f"{pdb_key}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        # Parse the restriction string into lists of positions and allowed amino acids
        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(pos_restrict_aatype, example["atom_array"])

        # Mark positions with restrictions
        restrict_pos_mask[i, abs_pos] = 1.0

        # Apply restrictions for each position
        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            # First, disallow all amino acids at this position
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            # Then allow only the specified amino acids
            for aa in allowed_aa:
                if aa in const.PROT_LETTER_TO_TOKEN:
                    allowed_aatype_mask[i, pos_idx, const.AF3_ENCODING.encode_aa(aa)] = 1.0
                else:
                    print(
                        f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_key} "
                        f"at position {pdb_pos[abs_pos.index(pos_idx)]}"
                    )

        if verbose:
            # Print a summary of the restrictions
            for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
                pos_str = pdb_pos[abs_pos.index(pos_idx)]
                print(f" * Position {pos_str}: Restricted to {allowed_aa}")

    if verbose:
        print("\n********************************************************\n")

    return restrict_pos_mask, allowed_aatype_mask


def parse_symmetry_pos_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> list[list[int]] | None:
    """
    Given a pos_constraint_df containing symmetry positions for each PDB, return a list of lists of absolute positions.
    Each inner list contains positions that should be sampled symmetrically.

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["symmetry_pos"]
    where each entry is a string in the format "A15,B15|A16,B16|A17,B17", or np.nan.
    Groups are separated by "|" and positions within a group are separated by ",".
    """
    if pos_constraint_df is None:
        if verbose:
            print("No symmetry positions specified.")
        return None

    # Cache symmetry positions by tied sampling id to avoid re-parsing.
    tied_sampling_id_cache = {}

    batch_symmetry_pos = []
    for i, example_id in enumerate(batch["example_id"]):
        if example_id not in pos_constraint_df.index:
            if verbose:
                print(f"{example_id}: No symmetry positions specified.")
            batch_symmetry_pos.append([])
            continue

        # Get symmetry positions from df.
        row = pos_constraint_df.loc[example_id]
        symmetry_pos_str = row.get("symmetry_pos", np.nan)

        if pd.isna(symmetry_pos_str):
            if verbose:
                print(f"{example_id}: No symmetry positions specified.")
            batch_symmetry_pos.append([])
            continue

        # Set up example.
        example = {k: v[i] for k, v in batch.items()}

        if verbose:
            print(f"{example_id}: Parsing symmetry positions {symmetry_pos_str}")

        # Parse the symmetry position string.
        tied_sampling_id = None
        if "tied_sampling_ids" in example:
            # Handle caching of symmetry positions by tied sampling id.
            tied_sampling_id = example["tied_sampling_ids"].item()
            if tied_sampling_id not in tied_sampling_id_cache:
                tied_sampling_id_cache[tied_sampling_id] = parse_symmetry_pos_str(
                    symmetry_pos_str, example["atom_array"]
                )
            symmetry_pos = tied_sampling_id_cache[tied_sampling_id]
        else:
            # No tied sampling ids, just parse the symmetry positions.
            symmetry_pos = parse_symmetry_pos_str(symmetry_pos_str, example["atom_array"])

        batch_symmetry_pos.append(symmetry_pos)
    return batch_symmetry_pos


def parse_fixed_pos_str(fixed_pos_str: str, atom_array: AtomArray) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A", "B1", "C10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A,B1,C10-25").
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        TensorType["k", int]: List of absolute indices to set to 1 in the masks.
    """
    chain_annotation = "chain_id"  # we use chain_id for fixing positions
    residue_index = atom_array.res_id[get_token_starts(atom_array)]
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match_with_residues = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        # Match pattern for just a chain ID, e.g., "A"
        match_chain_only = re.match(r"([A-Za-z])$", pos)

        if match_with_residues:
            chain_letter = match_with_residues.group(1)
            start_residue = int(match_with_residues.group(2))
            end_residue = int(match_with_residues.group(3)) if match_with_residues.group(3) else start_residue

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(
                    f"Chain ID {chain_letter} not found in chain annotation: {np.unique(atom_array.get_annotation(chain_annotation))}."
                )

            # For the given chain, create a mask for all residues in the desired range
            atomwise_range_mask = (
                (atom_array.get_annotation(chain_annotation) == chain_letter)
                & (atom_array.res_id >= start_residue)
                & (atom_array.res_id <= end_residue)
            )
            range_mask = apply_token_wise(atom_array, atomwise_range_mask, np.any)  # get per-token mask
            matching_indices = np.where(range_mask)[0]

            # Check that each residue in the requested range; warn if not found
            found_residues = set(residue_index[matching_indices].tolist())

            for r in range(start_residue, end_residue + 1):
                if r not in found_residues:
                    print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

            # Extend our fixed indices with whatever we did find
            fixed_indices.extend(matching_indices.tolist())
        elif match_chain_only:
            chain_letter = match_chain_only.group(1)

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(
                    f"Chain ID {chain_letter} not found in chain annotation: {np.unique(atom_array.get_annotation(chain_annotation))}."
                )

            # For the given chain, create a mask for all residues
            atomwise_chain_mask = atom_array.get_annotation(chain_annotation) == chain_letter
            chain_mask = apply_token_wise(atom_array, atomwise_chain_mask, np.any)
            matching_indices = np.where(chain_mask)[0]
            fixed_indices.extend(matching_indices.tolist())
        else:
            raise ValueError(f"Invalid position format: {pos}")

    return fixed_indices


def parse_fixed_pos_override_seq_str(
    override_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a fixed position sequence override string in the format "A26:A,A27:L" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and override amino acids (e.g., ["A", "L"]).

    Args:
        override_str (str): Comma-separated string of position overrides
                           in the format "<chain+residue>:<desired aatype>"
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        tuple: (pdb_pos, abs_pos, override_aatypes) - lists with corresponding entries
    """
    if not override_str or override_str.strip() == "":
        return [], [], []

    pdb_pos = []
    override_aatypes = []

    # Split by comma and process each override
    overrides = [o.strip() for o in override_str.split(",") if o.strip()]

    for override in overrides:
        # Split by colon to get position and override aatype
        parts = override.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid override format: {override}. Expected format: 'A26:A'")

        pos, aatype = parts[0].strip(), parts[1].strip()

        if len(aatype) != 1 or aatype not in const.PROT_LETTER_TO_TOKEN:
            raise ValueError(f"Invalid aatype: {aatype} in {override}. Expected single letter amino acid code.")

        pdb_pos.append(pos)
        override_aatypes.append(aatype)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, override_aatypes


def parse_pos_restrict_aatype_str(
    pos_restrict_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a position restriction string in the format "A26:AVG,A27:VG" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and allowed aatypes (e.g., ["AVG", "VG"]).

    Args:
        pos_restrict_str (str): Comma-separated string of position restrictions
                               in the format "<chain+residue>:<allowed aatypes>"
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        tuple: (pdb_pos, abs_pos, allowed_aatypes) - lists with corresponding entries
    """
    if not pos_restrict_str or pos_restrict_str.strip() == "":
        return [], [], []

    pdb_pos = []
    allowed_aatypes = []

    # Split by comma and process each restriction.
    restrictions = [r.strip() for r in pos_restrict_str.split(",") if r.strip()]

    for restriction in restrictions:
        # Split by colon to get position and allowed aatypes.
        parts = restriction.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid restriction format: {restriction}. Expected format: 'A26:AVG'")

        pos, aatypes = parts[0].strip(), parts[1].strip()
        pdb_pos.append(pos)
        allowed_aatypes.append(aatypes)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, allowed_aatypes


def parse_symmetry_pos_str(symmetry_pos_str: str, atom_array: AtomArray) -> list[list[int]]:
    """
    Parse a symmetry position string in the format "A15,B15|A16,B16|A17,B17" into a list of lists of absolute positions.
    Each group of positions denotes positions to be sampled symmetrically.
    """
    if not symmetry_pos_str or symmetry_pos_str.strip() == "":
        return []

    symmetry_pos = []
    for group in symmetry_pos_str.split("|"):
        symmetry_pos.append(parse_fixed_pos_str(group.strip(), atom_array))
    return symmetry_pos


def visualize_conditioning_sequences(
    atom_array: AtomArray,
    cond_mask: TensorType["n", int],
    asym_id: TensorType["n", int],
    asym_names: list[str],
) -> str:
    """
    Visualize the conditioning sequence for a given atom array.
    """
    chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_array)
    sequences = {}

    # Map from chain_name to asym_id.
    chain_names = [x.split("_")[0] for x in asym_names]  # for now, ignore transforms
    chain_name_to_asym_id = {chain_name: i for i, chain_name in enumerate(chain_names)}

    for chain_name, info in chain_info.items():
        sequence = info["processed_entity_canonical_sequence"]
        # Replace with "-" where cond_mask is 0
        chain_cond_mask = cond_mask[asym_id == chain_name_to_asym_id[chain_name]]
        sequence = "".join([x if chain_cond_mask[i] else "-" for i, x in enumerate(sequence)])
        sequences[chain_name] = sequence

    for chain_name, sequence in sequences.items():
        print(f"Chain {chain_name}: {sequence}")


def _validate_ensemble_alignment(batch: dict[str, TensorType["b ..."]]):
    """
    Validate that the alignment of the batch is correct.
    """
    if not (batch["residue_index"] == batch["residue_index"][0]).all().item():
        raise ValueError(
            "Residue index mismatch between decoys. If positions are not aligned, aggregation of potts "
            "parameters will be incorrect and will yield nonsensical results. If this was intentional, "
            "set ensemble_ignore_res_idx_mismatch=True."
        )
    if not (batch["asym_id"] == batch["asym_id"][0]).all().item():
        raise ValueError(
            "Chain ID mismatch between decoys. If positions are not aligned, aggregation of potts "
            "parameters will be incorrect and will yield nonsensical results. If this was intentional, "
            "set ensemble_ignore_res_idx_mismatch=True."
        )
