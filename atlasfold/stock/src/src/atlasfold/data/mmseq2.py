"""Default MMseqs2 commands"""

import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from atlasfold.data.fasta import write_fasta


def run_mmseqs2_cluster(
    sequences: list[tuple[str, str]] | dict[str, str],
    min_sequence_identity: float = 0.4,
    coverage: float = 0.8,
    coverage_mode: int = 0,
    sensitivity: float = 8.0,
    max_sequences: int | None = 1000,
    verbose: int = 3,
    print_cmd: bool = True,
    mmseqs2_exec: str = "mmseqs",
) -> dict[str, str]:
    """Cluster sequences using MMseqs2.

    Parameters
    ----------
    sequences : list[tuple[str, str]] | dict[str, str]
        List of tuples (ID, sequence) or a dictionary mapping IDs to sequences.
    min_sequence_identity : float, optional
        Minimum sequence identity for clustering, by default 0.4.
    coverage : float, optional
        Minimum coverage for clustering, by default 0.8.
    coverage_mode : int, optional
        Coverage mode for MMseqs2, by default 0.
    sensitivity : float, optional
        Sensitivity for MMseqs2 clustering, by default 8.0.
    max_sequences : int | None, optional
        Maximum number of sequences to cluster, by default 1000.
    verbose : int, optional
        Verbosity level for MMseqs2, by default 3.
    print_cmd : bool, optional
        Whether to print the MMseqs2 command, by default True.
    mmseqs2_exec : str, optional
        MMseqs2 executable command, by default "mmseqs".

    Returns
    -------
    cluster_map: dict[str, str]
        Mapping from sequence ID to cluster ID.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Write sequences to a temporary fasta file
        input_fasta_path = tmp_dir / "input.fasta"
        write_fasta(sequences, input_fasta_path)

        out_prefix = tmp_dir / "mmseq2Res"

        # Construct MMseqs2 command
        cmd_str = (
            f"{mmseqs2_exec} easy-cluster "
            f"{input_fasta_path} {out_prefix} {tmp_dir} "
            f"--min-seq-id {min_sequence_identity} "
            f"-c {coverage} --cov-mode {coverage_mode} "
            f"-s {sensitivity} "
            f"--dbtype 1 "
            f"-v {verbose} "
        )
        if max_sequences is not None:
            cmd_str += f"--max-seqs {max_sequences} "

        if print_cmd:
            print("Running MMseqs2 clustering with command:")
            print(cmd_str)

        # Execute the command
        subprocess.run(cmd_str, shell=True, check=True)

        # Read clustering output
        cluster_map: dict[str, str] = {}
        with open(tmp_dir / "mmseq2Res_cluster.tsv") as f:
            # Read the clustering results
            for line in f:
                cluster_id, seq_id = line.strip().split("\t")
                cluster_map[seq_id] = cluster_id

    return cluster_map


def run_mmseqs2_search(
    queries: list[tuple[str, str]] | dict[str, str],
    targets: list[tuple[str, str]] | dict[str, str],
    min_sequence_identity: float = 0.4,
    coverage: float = 0.8,
    coverage_mode: int = 2,
    sensitivity: float = 8.0,
    verbose: int = 3,
    print_cmd: bool = True,
    mmseqs2_exec: str = "mmseqs",
) -> dict[str, set[str]]:
    """Find homologous sequences using MMseqs2 search.

    Parameters
    ----------
    queries : list[tuple[str, str]] | dict[str, str]
        A list of tuples (ID, sequence) for query sequences
        or a dictionary mapping IDs to sequences.
    targets : list[tuple[str, str]] | dict[str, str]
        A list of tuples (ID, sequence) for target sequences
        or a dictionary mapping IDs to sequences.
    min_sequence_identity : float, optional
        Minimum sequence identity for clustering, by default 0.4.
    coverage : float, optional
        Minimum coverage for clustering, by default 0.8.
    coverage_mode : int, optional
        Coverage mode for MMseqs2, by default 2 (query coverage).
    sensitivity : float, optional
        Sensitivity for MMseqs2 clustering, by default 8.0.
    verbose : int, optional
        Verbosity level for MMseqs2, by default 3.
    print_cmd : bool, optional
        Whether to print the MMseqs2 command, by default True.
    mmseqs2_exec : str, optional
        MMseqs2 executable command, by default "mmseqs".

    Returns
    -------
    homolog_map: dict[str, set[str]]
        Mapping from query sequence ID to a set of homologous target sequence IDs.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        # Write sequences to a temporary fasta file
        query_fasta = tmp_dir / "query.fasta"
        write_fasta(queries, query_fasta)

        target_fasta = tmp_dir / "target.fasta"
        write_fasta(targets, target_fasta)

        max_seqs = len(targets)

        # out file
        out_tsv = tmp_dir / "mmseq2Res.tsv"

        # Construct MMseqs2 command
        cmd_str = (
            f"{mmseqs2_exec} easy-search "
            f"{query_fasta} {target_fasta} {out_tsv} {tmp_dir} "
            f"--min-seq-id {min_sequence_identity} "
            f"-c {coverage} --cov-mode {coverage_mode} "
            f"-s {sensitivity} "
            f"--max-seqs {max_seqs} "
            f"--dbtype 1 "
            f"--search-type 1 "
            f"-v {verbose} "
        )

        if print_cmd:
            print("Running MMseqs2 clustering with command:")
            print(cmd_str)

        # Execute the command
        subprocess.run(cmd_str, shell=True, check=True)

        homologs: dict[str, set[str]] = defaultdict(set)
        with open(out_tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                qid, tid, ident = parts[0], parts[1], float(parts[2])
                if ident >= min_sequence_identity:
                    homologs[qid].add(tid)

    return homologs
