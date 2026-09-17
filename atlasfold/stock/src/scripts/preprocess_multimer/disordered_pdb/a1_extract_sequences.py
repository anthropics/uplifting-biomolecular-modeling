"""Preprocess RCSB mmCIF files."""

import argparse
import logging
import multiprocessing
import os
import pathlib
from datetime import datetime

import gemmi
import msgpack
from tqdm import tqdm

from atlasfold.common import residue_constants
from atlasfold.data import fasta

logger = logging.getLogger(__name__)

DATE_END: datetime = datetime.fromisoformat("2021-09-30")


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
        help="Path to output directory for save sequences",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()
    return args


def parse_cif(cif_path: pathlib.Path) -> list[tuple[str, str]]:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    pdb_id = cif_path.stem.split(".")[0]

    # Read CIF file
    block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    raw_struct.assign_subchains()
    raw_struct.setup_entities()

    sequences: list[tuple[str, str]] = []
    seq_to_eid: dict[str, int] = {}
    for entity in raw_struct.entities:
        if not (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
        ):
            # Skip non-protein polymers
            continue
        subchain = entity.subchains[0]
        aas = []
        for res in raw_struct[0].get_subchain(subchain):
            aas.append(residue_constants.restype_3to1.get(res.name, "X"))
        sequence = "".join(aas)

        if sequence in seq_to_eid:
            # NOTE: OpenFold does not handle duplicate sequences well...
            continue
        entity_id = len(seq_to_eid) + 1
        seq_to_eid[sequence] = entity_id
        sequences.append((f"{pdb_id}_{entity_id}", sequence))
    return sequences


def worker_fn(cif_path: pathlib.Path) -> list[tuple[str, str]]:
    try:
        return parse_cif(cif_path)
    except Exception as e:
        print(f"Failed to process ({cif_path}): {e}")
        return []


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_multimer/"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_data_dir: pathlib.Path = args.data_dir / "rcsb_multimer/"
    metadata_path = train_data_dir / "manifest.msgpack"
    with open(metadata_path, "rb") as f:
        metadata = msgpack.unpack(f, raw=False)
    pdb_ids = set(m["id"] for m in metadata)

    cif_paths = sorted(cif_dir.rglob("*.cif"))
    cif_paths = [p for p in cif_paths if p.stem.split(".")[0] in pdb_ids]
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    all_sequences: list[tuple[str, str]] = []
    with multiprocessing.Pool(args.num_workers) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_fn, cif_paths),
            total=len(cif_paths),
            desc="Processing RCSB mmCIF files",
        )
        for seqs in pbar:
            all_sequences.extend(seqs)
            pbar.set_postfix({"Total sequences": len(all_sequences)})
    print("Processing completed.")

    # Print stats
    print(f"Total unique sequences extracted: {len(all_sequences)}")

    fasta_path = data_dir / "rcsb_sequences.fasta"
    all_sequences = sorted(set(all_sequences), key=lambda x: x[0])
    fasta.write_fasta(all_sequences, fasta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
