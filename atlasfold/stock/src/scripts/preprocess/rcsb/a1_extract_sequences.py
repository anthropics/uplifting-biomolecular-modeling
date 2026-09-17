"""Preprocess RCSB mmCIF files."""

import argparse
import logging
import multiprocessing
import os
import pathlib
from datetime import datetime
from typing import Any

import gemmi
from tqdm import tqdm

from atlasfold.common import ccd, residue_constants
from atlasfold.data import fasta

logger = logging.getLogger(__name__)

DATE_END: datetime = datetime.fromisoformat("2020-05-01")


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


def get_first_value(block: gemmi.cif.Block, tag: str, cast: type = str) -> Any | None:
    values = block.find_values(tag)
    if len(values) > 0:
        try:
            return cast(values[0])
        except Exception:
            return None
    return None


def ccd_to_sequence(ccd_sequence: list[str]) -> str:
    def convert_ccd_to_aa(v: str) -> str:
        aa = ccd.CCD_NAME_TO_ONE_LETTER.get(v, "X")
        return aa if aa in residue_constants.restype_1to3 else "X"

    return "".join(convert_ccd_to_aa(v) for v in ccd_sequence)


def parse_cif(cif_path: pathlib.Path) -> list[tuple[str, str]]:
    """Parse a CIF file and return a gemmi.cif.Document object."""
    # Read CIF file
    if cif_path.suffix == ".gz":
        doc: gemmi.cif.Document = gemmi.cif.read(str(cif_path))
    else:
        doc: gemmi.cif.Document = gemmi.cif.read_file(str(cif_path))
    block: gemmi.cif.Block = doc[0]

    # PDB ID
    pdb_id = get_first_value(block, "_entry.id")
    assert pdb_id is not None, "PDB ID is missing in metadata."

    # Release Date
    rev_dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    release_date = min(rev_dates) if rev_dates else None
    assert release_date is not None, "Release date is missing in metadata."
    # Filter by date
    release_date = datetime.fromisoformat(release_date)
    if not release_date <= DATE_END:
        return []

    # Prepare gemmi structure
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)

    sequences: list[tuple[str, str]] = []
    for entity in raw_struct.entities:
        if not (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
        ):
            # Skip non-protein polymers
            continue
        entity_id = entity.name
        sequence = ccd_to_sequence(entity.full_sequence)
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
    data_dir: pathlib.Path = args.data_dir / "rcsb/"

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
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
