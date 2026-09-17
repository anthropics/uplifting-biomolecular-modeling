"""Construct the RCSB protein-multimer validation split.

This mirrors the KFold RCSB validation construction, specialized to protein
multimers. Candidate structures are expected under ``rcsb_multimer_val/npz``
after running ``b1_process.py --split val``.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import pathlib
from collections import defaultdict
from typing import NamedTuple, TypeVar

import msgpack
import numpy as np
from tqdm import tqdm

from atlasfold.data.fasta import read_fasta
from atlasfold.data.mmseq2 import run_mmseqs2_cluster, run_mmseqs2_search
from atlasfold.train.multimer.dataset import MultimerDataPipeline

_T = TypeVar("_T")

VERBOSE = 0
MIN_RESIDUES = 16
MAX_RESIDUES = 1536
MAX_CHAINS = 20
SEQUENCE_IDENTITY_THRESHOLD = 0.40
FINAL_VALIDATION_SET_SIZE = 512


class Seq(NamedTuple):
    pdb_id: str
    entity_id: int
    sequence: str

    @property
    def id(self) -> str:
        return f"{self.pdb_id}_{self.entity_id}"


class Interface(NamedTuple):
    seq1: Seq
    seq2: Seq

    @property
    def pdb_id(self) -> str:
        return self.seq1.pdb_id

    @property
    def id(self) -> str:
        eid1, eid2 = norm_key(self.seq1.entity_id, self.seq2.entity_id)
        return f"{self.pdb_id}_{eid1}:{eid2}"


class EntryData(NamedTuple):
    pdb_id: str
    npz_path: pathlib.Path
    metadata: dict
    chains: list[Seq]
    interfaces: list[Interface]
    num_chains: int
    num_residues: int


def norm_key(key1: _T, key2: _T) -> tuple[_T, _T]:
    """Return a normalized tuple of two sortable keys."""
    return (key1, key2) if key1 <= key2 else (key2, key1)


def get_rng(key: str) -> np.random.Generator:
    """Get a deterministic random number generator seeded by key."""
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed)


def parse_entity_id(seq_id: str) -> tuple[str, int]:
    """Parse ``{pdb_id}_{entity_id}`` sequence ids."""
    pdb_id, entity_id = seq_id.rsplit("_", 1)
    return pdb_id.lower(), int(entity_id)


def load_train_sequences(train_dir: pathlib.Path) -> list[Seq]:
    """Load training protein entity sequences from the multimer FASTA."""
    fasta_path = train_dir / "rcsb_sequences.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"Training sequence FASTA not found: {fasta_path}")

    train_sequences: list[Seq] = []
    for seq_id, sequence in read_fasta(fasta_path):
        pdb_id, entity_id = parse_entity_id(seq_id)
        train_sequences.append(Seq(pdb_id, entity_id, sequence))
    return train_sequences


def read_entry(npz_path: pathlib.Path) -> EntryData:
    """Read one processed multimer entry and extract protein interfaces."""
    metadata_path = npz_path.with_suffix(".json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    compl = MultimerDataPipeline.load(npz_path)

    pdb_id = metadata["id"].lower()
    chain_metadatas = metadata["chains"]
    if compl.num_chains != len(chain_metadatas):
        raise ValueError(
            f"{pdb_id} has {compl.num_chains} NPZ chains but "
            f"{len(chain_metadatas)} metadata chains."
        )

    entity_sequences: dict[int, Seq] = {}
    for chain, chain_metadata in zip(compl.chains, chain_metadatas, strict=True):
        entity_id = int(chain_metadata["entity_id"])
        if entity_id not in entity_sequences:
            entity_sequences[entity_id] = Seq(pdb_id, entity_id, chain.sequence)
    chains = sorted(entity_sequences.values(), key=lambda seq: seq.id)

    interfaces: list[Interface] = []
    visited: set[tuple[int, int]] = set()
    for interface_metadata in metadata.get("interfaces", []):
        chain_i, chain_j = interface_metadata["chain_ids"]
        entity_i = int(chain_metadatas[chain_i]["entity_id"])
        entity_j = int(chain_metadatas[chain_j]["entity_id"])
        key = norm_key(entity_i, entity_j)
        if key in visited:
            continue
        visited.add(key)
        interfaces.append(
            Interface(entity_sequences[entity_i], entity_sequences[entity_j])
        )

    num_residues = sum(int(chain["num_residues"]) for chain in chain_metadatas)
    metadata["id"] = pdb_id
    metadata["num_chains"] = len(chain_metadatas)
    metadata["num_residues"] = num_residues

    return EntryData(
        pdb_id=pdb_id,
        npz_path=npz_path,
        metadata=metadata,
        chains=chains,
        interfaces=interfaces,
        num_chains=len(chain_metadatas),
        num_residues=num_residues,
    )


def get_protein_homologs(
    queries: list[Seq],
    targets: list[Seq],
    mmseqs: str,
    sequence_identity: float = SEQUENCE_IDENTITY_THRESHOLD,
) -> dict[str, set[str]]:
    """Return training PDB ids with homologs for each query sequence."""
    print(
        f"Getting protein homologs with sequence identity >= {sequence_identity:.0%}..."
    )
    if len(queries) == 0:
        return {}
    if len(targets) == 0:
        return {seq.id: set() for seq in queries}

    seq_to_query_ids: dict[str, set[str]] = defaultdict(set)
    for seq in queries:
        seq_to_query_ids[seq.sequence].add(seq.id)
    uniq_queries: dict[str, str] = {
        f"query-{i}": seq for i, seq in enumerate(sorted(seq_to_query_ids))
    }

    seq_to_target_pdbs: dict[str, set[str]] = defaultdict(set)
    for seq in targets:
        seq_to_target_pdbs[seq.sequence].add(seq.pdb_id)
    uniq_targets: dict[str, str] = {
        f"target-{i}": seq for i, seq in enumerate(sorted(seq_to_target_pdbs))
    }

    homolog_out = run_mmseqs2_search(
        uniq_queries,
        uniq_targets,
        min_sequence_identity=sequence_identity,
        coverage=0.8,
        coverage_mode=2,
        verbose=VERBOSE,
        print_cmd=(VERBOSE > 0),
        mmseqs2_exec=mmseqs,
    )

    results: dict[str, set[str]] = {seq.id: set() for seq in queries}
    for query_id, target_ids in homolog_out.items():
        query_sequence = uniq_queries[query_id]
        for target_id in target_ids:
            target_pdbs = seq_to_target_pdbs[uniq_targets[target_id]]
            for seq_id in seq_to_query_ids[query_sequence]:
                results[seq_id].update(target_pdbs)

    for seq in queries:
        if seq.sequence in seq_to_target_pdbs:
            results[seq.id].update(seq_to_target_pdbs[seq.sequence])

    n_low_homology = sum(1 for pdb_ids in results.values() if len(pdb_ids) == 0)
    print(f"Total query protein chains: {len(results)}")
    print(f"Total target protein chains: {len(targets)}")
    print(f"Low homology protein chains: {n_low_homology}")
    return results


def run_protein_clustering(
    all_sequences: list[Seq],
    mmseqs: str,
    sequence_identity: float = SEQUENCE_IDENTITY_THRESHOLD,
) -> dict[str, str]:
    """Cluster protein sequences, using exact clusters for very short chains."""
    sequence_to_repr_id: dict[str, str] = {}
    short_sequence_to_repr_id: dict[str, str] = {}
    for seq in sorted(all_sequences, key=lambda s: s.id):
        if len(seq.sequence) >= 10:
            sequence_to_repr_id.setdefault(seq.sequence, seq.id)
        else:
            short_sequence_to_repr_id.setdefault(seq.sequence, seq.id)

    uniq_sequences = sorted(
        (repr_id, seq) for seq, repr_id in sequence_to_repr_id.items()
    )
    if len(uniq_sequences) > 0:
        cluster_map = run_mmseqs2_cluster(
            uniq_sequences,
            min_sequence_identity=sequence_identity,
            coverage=0.8,
            coverage_mode=0,
            verbose=VERBOSE,
            print_cmd=(VERBOSE > 0),
            mmseqs2_exec=mmseqs,
        )
    else:
        cluster_map = {}

    sequence_to_cluster_id = {
        sequence: cluster_map[repr_id]
        for sequence, repr_id in sequence_to_repr_id.items()
    }
    sequence_to_cluster_id |= short_sequence_to_repr_id
    return {seq.id: sequence_to_cluster_id[seq.sequence] for seq in all_sequences}


def filter_multimer_interfaces(
    all_interfaces: list[Interface],
    train_sequences: list[Seq],
    mmseqs: str,
    sequence_identity: float,
) -> list[Interface]:
    """Filter, cluster, and sample protein-protein interfaces."""
    print("=" * 50)
    print("Protein-Protein Interface Filtering")
    print("Total interfaces before filtering:", len(all_interfaces))

    all_sequences: list[Seq] = []
    collected_ids: set[str] = set()
    for interface in sorted(all_interfaces, key=lambda iface: iface.id):
        for seq in interface:
            if seq.id not in collected_ids:
                collected_ids.add(seq.id)
                all_sequences.append(seq)

    print("\nStage 1: Homology search...")
    homologs = get_protein_homologs(
        all_sequences,
        train_sequences,
        mmseqs=mmseqs,
        sequence_identity=sequence_identity,
    )

    print("\nStage 2: Homology filtering of interfaces...")
    filtered_interfaces = []
    for interface in tqdm(all_interfaces, desc="Homology Filtering"):
        train_pdb1 = homologs[interface.seq1.id]
        train_pdb2 = homologs[interface.seq2.id]
        if len(train_pdb1 & train_pdb2) == 0:
            filtered_interfaces.append(interface)
    print(
        f"Total interfaces after homology filtering: {len(filtered_interfaces)} "
        f"out of {len(all_interfaces)}"
    )

    print("\nStage 3: Clustering interfaces...")
    all_sequences = []
    collected_ids = set()
    for interface in sorted(filtered_interfaces, key=lambda iface: iface.id):
        for seq in interface:
            if seq.id not in collected_ids:
                collected_ids.add(seq.id)
                all_sequences.append(seq)
    clusters = run_protein_clustering(
        all_sequences,
        mmseqs=mmseqs,
        sequence_identity=sequence_identity,
    )

    interface_clusters: dict[str, list[Interface]] = defaultdict(list)
    for interface in filtered_interfaces:
        cluster_id = "__".join(
            norm_key(clusters[interface.seq1.id], clusters[interface.seq2.id])
        )
        interface_clusters[cluster_id].append(interface)

    sampled_interfaces: list[Interface] = []
    for cluster_id, interfaces in interface_clusters.items():
        interfaces.sort(key=lambda iface: iface.id)
        rng = get_rng(cluster_id)
        sampled_interfaces.append(interfaces[rng.integers(len(interfaces))])
    print(f"Total interfaces after clustering: {len(sampled_interfaces)}")
    return sampled_interfaces


def mark_low_homology(
    entries: list[EntryData],
    train_sequences: list[Seq],
    mmseqs: str,
    sequence_identity: float,
) -> list[dict]:
    """Mark selected validation chains and interfaces with low-homology flags."""
    all_sequences: list[Seq] = []
    collected_ids: set[str] = set()
    for entry in entries:
        for seq in entry.chains:
            if seq.id not in collected_ids:
                collected_ids.add(seq.id)
                all_sequences.append(seq)

    homologs = get_protein_homologs(
        all_sequences,
        train_sequences,
        mmseqs=mmseqs,
        sequence_identity=sequence_identity,
    )

    metadatas: list[dict] = []
    for entry in entries:
        seq_by_entity = {seq.entity_id: seq for seq in entry.chains}
        metadata = entry.metadata
        metadata["num_chains"] = entry.num_chains
        metadata["num_residues"] = entry.num_residues

        for chain_metadata in metadata["chains"]:
            seq = seq_by_entity[int(chain_metadata["entity_id"])]
            if len(homologs[seq.id]) == 0:
                chain_metadata["is_low_homology"] = True

        for interface_metadata in metadata.get("interfaces", []):
            chain_i, chain_j = interface_metadata["chain_ids"]
            entity_i = int(metadata["chains"][chain_i]["entity_id"])
            entity_j = int(metadata["chains"][chain_j]["entity_id"])
            seq_i = seq_by_entity[entity_i]
            seq_j = seq_by_entity[entity_j]
            if len(homologs[seq_i.id] & homologs[seq_j.id]) == 0:
                interface_metadata["is_low_homology"] = True

        metadatas.append(metadata)
    return metadatas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct RCSB protein-multimer validation IDs and manifest."
    )
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to data root, rcsb_multimer, or rcsb_multimer_val.",
    )
    parser.add_argument(
        "--mmseqs",
        type=str,
        default="mmseqs",
        help="MMseqs2 executable.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=len(os.sched_getaffinity(0)),
        help="Number of parallel workers used to read candidate entries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = args.data_dir / "rcsb_multimer"
    val_dir = args.data_dir / "rcsb_multimer_val"
    npz_dir = val_dir / "npz"

    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Validation residue cap: {MAX_RESIDUES}")
    print(f"Validation chain cap: {MAX_CHAINS}")

    train_sequences = load_train_sequences(train_dir)
    print(f"Loaded {len(train_sequences)} training protein entity sequences.")

    npz_files = sorted(npz_dir.rglob("*.npz"))
    print(f"Total validation candidate NPZ files found: {len(npz_files)}")
    with multiprocessing.Pool(args.num_workers) as pool:
        entries = list(
            tqdm(
                pool.imap_unordered(read_entry, npz_files, chunksize=10),
                total=len(npz_files),
                desc="Reading validation candidates",
            )
        )

    entry_by_id = {entry.pdb_id: entry for entry in entries}
    candidate_entries = [
        entry
        for entry in entries
        if MIN_RESIDUES <= entry.num_residues <= MAX_RESIDUES
        and 2 <= entry.num_chains <= MAX_CHAINS
        and len(entry.interfaces) > 0
    ]

    interfaces: list[Interface] = []
    for entry in candidate_entries:
        interfaces.extend(entry.interfaces)
    interfaces.sort(key=lambda iface: iface.id)

    print(f"Total entries read: {len(entries)}")
    print(f"Candidate entries after size/interface filtering: {len(candidate_entries)}")
    print(f"Total protein-protein interfaces collected: {len(interfaces)}")

    sampled_interfaces = filter_multimer_interfaces(
        all_interfaces=interfaces,
        train_sequences=train_sequences,
        mmseqs=args.mmseqs,
        sequence_identity=SEQUENCE_IDENTITY_THRESHOLD,
    )

    sampled_ids = sorted({interface.pdb_id for interface in sampled_interfaces})
    if len(sampled_ids) > FINAL_VALIDATION_SET_SIZE:
        sampled_indices = get_rng("final").choice(
            len(sampled_ids), size=FINAL_VALIDATION_SET_SIZE, replace=False
        )
        val_ids = [sampled_ids[idx] for idx in sorted(sampled_indices)]
    else:
        val_ids = sampled_ids

    print("\nValidation Set Final Summary")
    print(f"Sampled interfaces: {len(sampled_interfaces)}")
    print(f"Sampled entries: {len(sampled_ids)}")
    print(f"Final entries: {len(val_ids)}")

    val_dir.mkdir(parents=True, exist_ok=True)
    val_ids_path = val_dir / "validation_ids.txt"
    with open(val_ids_path, "w") as f:
        for pdb_id in val_ids:
            f.write(f"{pdb_id}\n")
    print(f"Validation set PDB IDs saved to: {val_ids_path}")

    selected_entries = [entry_by_id[pdb_id] for pdb_id in val_ids]
    metadatas = mark_low_homology(
        selected_entries,
        train_sequences=train_sequences,
        mmseqs=args.mmseqs,
        sequence_identity=SEQUENCE_IDENTITY_THRESHOLD,
    )

    manifest_msgpack_path = val_dir / "manifest.msgpack"
    with open(manifest_msgpack_path, "wb") as f:
        msgpack.pack(metadatas, f, use_bin_type=True)
    print(f"Saved manifest (msgpack) to {manifest_msgpack_path}")

    manifest_json_path = val_dir / "manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(metadatas, f, indent=2)
    print(f"Saved manifest (json) to {manifest_json_path}")


if __name__ == "__main__":
    main()
