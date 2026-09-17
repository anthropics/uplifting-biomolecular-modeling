"""Cluster accepted disordered-PDB AF2 sequences with MMseqs2."""

import argparse
import pathlib

from _common import accepted_ids, dataset_dir

from atlasfold.data.fasta import read_fasta
from atlasfold.data.mmseq2 import run_mmseqs2_cluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--mmseqs", type=str, default="mmseqs")
    parser.add_argument(
        "--source_clusters",
        type=pathlib.Path,
        help=(
            "Project an existing sequence-compatible cluster CSV instead of "
            "rerunning MMseqs2."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = dataset_dir(args.data_dir)
    accepted = accepted_ids(args.data_dir)
    records = [
        record
        for record in read_fasta(output_dir / "unique_sequences.fasta")
        if record[0] in accepted
    ]
    if not records:
        raise ValueError("No accepted predictions found in gdt_report.csv")
    if args.source_clusters is not None:
        sequence_to_cluster: dict[str, str] = {}
        with args.source_clusters.open() as handle:
            next(handle)
            for line in handle:
                _, cluster_id, sequence = line.rstrip("\n").split(",", 2)
                # Match the historical ESMFold manifest construction: when the
                # source CSV assigns duplicate sequences to different cluster
                # representatives, the last row deterministically wins.
                sequence_to_cluster[sequence] = cluster_id
        id_to_cluster = {}
        for entity_id, sequence in records:
            try:
                id_to_cluster[entity_id] = sequence_to_cluster[sequence]
            except KeyError as error:
                raise KeyError(f"No source cluster for {entity_id}") from error
    else:
        id_to_cluster = run_mmseqs2_cluster(
            records,
            min_sequence_identity=0.4,
            coverage=0.8,
            coverage_mode=0,
            verbose=1,
            print_cmd=True,
            mmseqs2_exec=args.mmseqs,
        )
    id_to_sequence = dict(records)
    path = output_dir / "clusters.csv"
    with path.open("w") as handle:
        handle.write("entity_id,cluster_id,sequence\n")
        for entity_id, cluster_id in sorted(id_to_cluster.items()):
            handle.write(f"{entity_id},{cluster_id},{id_to_sequence[entity_id]}\n")
    print(f"Saved {len(set(id_to_cluster.values()))} clusters to {path}")


if __name__ == "__main__":
    main()
