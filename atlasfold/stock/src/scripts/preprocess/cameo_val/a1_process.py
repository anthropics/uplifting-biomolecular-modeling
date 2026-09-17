"""Preprocess CAMEO validation PDB entries"""

import argparse
import json
import logging
import pathlib

import gemmi
from tqdm import tqdm

from atlasfold.common import metadata, protein
from atlasfold.data import cif_factory
from atlasfold.train.monomer.dataset import DataPipeline

# Data filtering criteria
logger = logging.getLogger(__name__)
SUCCESS = 0
FAILED = 1


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RCSB mmCIF files.")
    parser.add_argument(
        "--cif_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the `mmCIF/` directory from RCSB.",
    )
    parser.add_argument(
        "--target_id",
        type=pathlib.Path,
        default=pathlib.Path("./assets/cameo_val_id.txt"),
        help="Path to the text file containing target chain IDs.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    args = parser.parse_args()
    return args


def parse_cif(
    target_chain_id: str,
    cif_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> int:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    # Read CIF file
    doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
    block: gemmi.cif.Block = doc[0]
    exp_record = cif_factory.read_experiment_record(block)

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    cif_factory.clean_up_gemmi_structure(raw_struct)

    # Extract name and target chain ID
    target_auth_id = target_chain_id.split("_")[-1]

    c: protein.Protein
    m: metadata.Metadata
    for c, ids in cif_factory.get_protein_chains(raw_struct):
        label_id = ids["label_id"]
        auth_id = ids["auth_id"]
        entity_id = ids["entity_id"]
        if auth_id != target_auth_id:
            continue

        c.name = target_chain_id
        m = metadata.Metadata(
            id=target_chain_id,
            label_asym_id=label_id,
            auth_asym_id=auth_id,
            entity_id=entity_id,
            num_residues=len(c),
            exp=exp_record,
        )
        break
    else:
        logger.warning(f"Target chain {target_chain_id} not found in {cif_path}.")
        return FAILED

    # Save each chain as a separate NPZ file
    npz_path = out_dir / f"{c.name}.npz"
    DataPipeline.save(c, npz_path)
    json_path = out_dir / f"{c.name}.json"
    m_dict = m.to_dict()
    with open(json_path, "w") as f:
        json.dump(m_dict, f)

    return SUCCESS


def worker_fn(args: tuple[str, pathlib.Path], output_dir: pathlib.Path):
    target_chain_id, cif_path = args
    try:
        return parse_cif(target_chain_id, cif_path, output_dir)
    except Exception as e:
        logger.error(f"Failed to process ({target_chain_id}): {e}")
        return FAILED, 0


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir / "cameo_val"

    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.target_id) as f:
        targets = [line.strip() for line in f if line.strip()]

    # Prepare tasks
    flags = []
    for target_id in tqdm(targets, desc="Processing CIF files"):
        pdb_id, chain_id = target_id.split("_")
        cif_path = cif_dir / pdb_id[1:3] / f"{pdb_id}.cif.gz"
        try:
            flag = parse_cif(target_id, cif_path, out_dir)
        except Exception as e:
            flag = FAILED
            logger.error(f"Failed to process ({target_id}): {e}")
        flags.append(flag)
    logger.info("Processing completed.")

    # Print stats
    logger.info(
        "Processing statistics:\n"
        f"  Total files processed: {len(flags)}\n"
        f"  Successfully processed: {flags.count(SUCCESS)}\n"
        f"  Failed to process: {flags.count(FAILED)}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
