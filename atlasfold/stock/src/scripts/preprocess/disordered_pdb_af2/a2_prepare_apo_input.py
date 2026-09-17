"""Deduplicate disordered-PDB sequences and assign stable prediction IDs."""

import argparse
import pathlib

from _common import dataset_dir

from atlasfold.data.fasta import read_fasta, write_fasta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = dataset_dir(args.data_dir)
    records = read_fasta(output_dir / "rcsb_sequences.fasta")
    sequences = sorted({sequence for _, sequence in records}, key=lambda x: (len(x), x))
    unique_records = [
        (f"seq_{index}", sequence) for index, sequence in enumerate(sequences)
    ]
    write_fasta(unique_records, output_dir / "unique_sequences.fasta")
    print(f"Saved {len(unique_records)} unique sequences")


if __name__ == "__main__":
    main()
