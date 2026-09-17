"""Prepare unique ESMFold inputs for the disordered-PDB dataset."""

import argparse
import logging
import pathlib

from atlasfold.data.fasta import read_fasta, write_fasta

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to output directory for save sequences",
    )
    args = parser.parse_args()
    return args


def main():
    """Main function to process RCSB mmCIF files"""
    args = parse_args()

    data_dir: pathlib.Path = args.data_dir / "disordered_pdb_esm/"

    # Read FASTA
    fasta_path = data_dir / "rcsb_sequences.fasta"
    sequences: list[tuple[str, str]] = read_fasta(fasta_path)

    # Extract unique sequences
    unique_fasta_path = data_dir / "unique_sequences.fasta"
    unique_sequences = sorted(set(seq for _, seq in sequences))
    unique_sequences.sort(key=lambda s: (len(s), s))
    unique_sequences = [(f"seq_{i}", seq) for i, seq in enumerate(unique_sequences)]
    write_fasta(unique_sequences, unique_fasta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
