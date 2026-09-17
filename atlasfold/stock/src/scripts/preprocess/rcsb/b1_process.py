"""Preprocess RCSB mmCIF files."""

import argparse
import functools
import json
import logging
import multiprocessing
import os
import pathlib
from datetime import datetime

import gemmi
import numpy as np
from tqdm import tqdm

from atlasfold.common import metadata, protein
from atlasfold.data import cif_factory
from atlasfold.train.monomer.dataset import DataPipeline

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
NO_PROTEIN_FILTERED = 4

# Data filtering criteria
DATE_END: datetime = datetime.fromisoformat("2020-05-01")
MAX_RESOLUTION: float = 9.0
MIN_LENGTH: int = 16

logger = logging.getLogger(__name__)

GLOBAL_CLUSTER_MAPPING: dict[str, str] = {}


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
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for processed .npz files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


# =================================================
# Helper functions for data filtering
# ==================================================
def validate_sequence(prot: protein.Protein) -> bool:
    # Check any single amino acid accounts for more than 80% of the sequence
    sequence = prot.sequence
    aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
    if any(count / len(sequence) > 0.8 for count in aa_counts.values()):
        logger.debug(
            f"{prot.name} marked invalid due to "
            f"overrepresentation of a single amino acid."
        )
        return False
    return True


def validate_geometry(prot: protein.Protein) -> bool:
    """Check if a polymer chain is valid."""
    ca_coords = prot.coordinates[:, 1, :]
    is_resolved = np.isfinite(ca_coords).all(axis=-1)
    n_resolved = np.sum(is_resolved)
    if n_resolved < MIN_LENGTH:
        logger.debug(f"{prot.name} marked invalid due to insufficient resolved CA atoms.")
        return False
    left = ca_coords[:-1]
    right = ca_coords[1:]
    dists = np.linalg.norm(left - right, axis=-1)
    if np.any(dists > 10.0):
        logger.debug(f"{prot.name} marked invalid due to CA trace discontinuity.")
        return False
    return True


def parse_cif(
    name: str,
    cif_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> tuple[int, int]:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    global GLOBAL_CLUSTER_MAPPING
    assert GLOBAL_CLUSTER_MAPPING, "Cluster mapping is not initialized in worker."

    # Read CIF file
    doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    exp_record = cif_factory.read_experiment_record(block)

    # Filter by date
    release_date = datetime.fromisoformat(exp_record.release_date)
    if not release_date <= DATE_END:
        return DATE_FILTERED, 0

    # Filter by resolution
    resolution = exp_record.resolution
    if resolution is not None and resolution > MAX_RESOLUTION:
        return RESOLUTION_FILTERED, 0

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)

    has_protein = False
    for entity in raw_struct.entities:
        if (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
            and len(entity.full_sequence) >= MIN_LENGTH
        ):
            has_protein = True
            break
    if not has_protein:
        return NO_PROTEIN_FILTERED, 0

    cif_factory.clean_up_gemmi_structure(raw_struct)

    chains: list[tuple[protein.Protein, metadata.Metadata]] = []
    for c, ids in cif_factory.get_protein_chains(raw_struct):
        label_id = ids["label_id"]
        auth_id = ids["auth_id"]
        entity_id = ids["entity_id"]
        c.name = f"{name}_{label_id}"

        # Get cluster ID from entity_id
        key = f"{name}_{entity_id}".upper()
        cluster_id = GLOBAL_CLUSTER_MAPPING.get(key)
        if cluster_id is None:
            logger.warning(f"Cluster ID not found for entity_id {entity_id} in {name}.")
            continue

        m = metadata.Metadata(
            id=f"{name}_{label_id}",
            label_asym_id=label_id,
            auth_asym_id=auth_id,
            entity_id=entity_id,
            num_residues=len(c),
            cluster_id=cluster_id,
            exp=exp_record,
        )
        chains.append((c, m))

    # Check sequence
    chains = [(c, m) for (c, m) in chains if validate_sequence(c)]

    # Check geometry
    chains = [(c, m) for (c, m) in chains if validate_geometry(c)]

    # Save each chain as a separate NPZ file
    for c, m in chains:
        npz_path = out_dir / f"{c.name}.npz"
        DataPipeline.save(c, npz_path)
        json_path = out_dir / f"{c.name}.json"
        m_dict = m.to_dict()
        with open(json_path, "w") as f:
            json.dump(m_dict, f)

    return SUCCESS, len(chains)


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
):
    # Output path
    pdb_id = cif_path.name.split(".")[0].lower()
    out_dir = output_dir / pdb_id[1:3] / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return parse_cif(pdb_id, cif_path, out_dir)
    except Exception as e:
        logger.error(f"Failed to process ({pdb_id}): {e}")
        return FAILED, 0


def worker_init(cluster_path: pathlib.Path):
    global GLOBAL_CLUSTER_MAPPING
    cluster_mapping = {}
    with open(cluster_path) as f:
        next(f)  # skip header
        for line in f:
            entity_id, cluster_id, _ = line.strip().split(",", 2)
            cluster_mapping[entity_id.upper()] = cluster_id.upper()
    GLOBAL_CLUSTER_MAPPING = cluster_mapping


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir / "rcsb/"

    # Read cluster mapping
    cluster_mapping_path = data_dir / "rcsb_clusters.csv"
    assert cluster_mapping_path.exists(), (
        f"Cluster mapping file not found: {cluster_mapping_path}"
    )
    cluster_mapping: dict[str, str] = {}
    with open(cluster_mapping_path) as f:
        next(f)  # skip header
        for line in f:
            entity_id, cluster_id, _ = line.strip().split(",", 2)
            cluster_mapping[entity_id.upper()] = cluster_id.upper()

    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare partial function for multiprocessing
    worker_wrapped = functools.partial(worker_fn, output_dir=out_dir)

    cif_paths = list(cif_dir.rglob("*.cif.gz"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    flags: list[int] = []
    n_total = 0
    with multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=(cluster_mapping_path,),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_wrapped, cif_paths, chunksize=100),
            total=len(cif_paths),
            desc="Processing RCSB mmCIF files",
        )
        for flag, n_success in pbar:
            n_total += n_success
            flags.append(flag)
            pbar.set_postfix({"Total Chains": n_total})
    print("Processing completed.")

    # Print stats
    print("Processing statistics:")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Failed to process: {flags.count(FAILED)}")
    print(f"  Date filtered: {flags.count(DATE_FILTERED)}")
    print(f"  Resolution filtered: {flags.count(RESOLUTION_FILTERED)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
