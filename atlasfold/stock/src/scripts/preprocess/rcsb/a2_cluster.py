"""Cluster RCSB training set sequences using MMseqs2."""

import argparse
import pathlib

from atlasfold.data.fasta import read_fasta
from atlasfold.data.mmseq2 import run_mmseqs2_cluster


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster training set sequences using MMseqs2.",
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to working directory.",
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default="mmseqs",
        help="MMseqs2 executable.",
    )
    args = parser.parse_args()

    return args


def main():
    """Main function to construct RCSB training set."""
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "rcsb/"

    # Step 1. Extract sequences from npz files
    fasta_path = data_dir / "rcsb_sequences.fasta"
    sequences: list[tuple[str, str]] = read_fasta(fasta_path)
    print(f"Extracted {len(sequences)} sequences from {fasta_path}")

    # Step 2. Cluster sequences using MMseqs2
    print("Starting sequence clustering...")
    # Perform clustering using MMseqs2
    print()
    entity_id_to_repr_id: dict[str, str] = run_mmseqs2_cluster(
        sequences,
        min_sequence_identity=0.4,
        coverage=0.8,
        coverage_mode=0,
        verbose=1,
        print_cmd=True,
        mmseqs2_exec=args.mmseqs,
    )
    print("Total clusters: ", len(set(entity_id_to_repr_id.values())))

    entity_id_to_sequence: dict[str, str] = dict(sequences)

    # Step 3. Save cluster output
    cluster_output_path = data_dir / "rcsb_clusters.csv"
    with open(cluster_output_path, "w") as f:
        f.write("entity_id,cluster_id,sequence\n")
        for entity_id, cluster_id in sorted(entity_id_to_repr_id.items()):
            seq = entity_id_to_sequence[entity_id]
            f.write(f"{entity_id},{cluster_id},{seq}\n")


if __name__ == "__main__":
    main()
