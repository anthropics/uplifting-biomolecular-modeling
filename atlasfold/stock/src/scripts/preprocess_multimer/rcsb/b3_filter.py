"""Filter the multimer RCSB manifest for downstream training subsets."""

import argparse
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any

import msgpack


def parse_args():
    parser = argparse.ArgumentParser(description="Filter the multimer RCSB manifest.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="manifest_confidence",
        help="Output manifest prefix (without .msgpack).",
    )
    parser.add_argument(
        "--min_resolution",
        type=float,
        default=0.1,
        help="Minimum resolution threshold for filtering structures.",
    )
    parser.add_argument(
        "--max_resolution",
        type=float,
        default=3.0,
        help="Maximum resolution threshold for filtering structures.",
    )
    parser.add_argument(
        "--max_release_date",
        type=str,
        default="2021-09-30",
        help="Latest allowed release date in ISO format. Use 'none' to disable.",
    )
    parser.add_argument(
        "--min_chains",
        type=int,
        default=1,
        help="Minimum number of chains in a complex.",
    )
    parser.add_argument(
        "--max_chains",
        type=int,
        default=20,
        help="Maximum number of chains after AF3 closest-chain extraction.",
    )
    parser.add_argument(
        "--min_residues",
        type=int,
        default=4,
        help="Minimum number of residues required for every chain.",
    )
    parser.add_argument(
        "--require_interface",
        action="store_true",
        help="Keep only complexes with at least one detected chain-chain interface.",
    )
    parser.add_argument(
        "--recompute_cluster_size",
        action="store_true",
        help="Recompute chain/interface cluster sizes after filtering.",
    )
    args = parser.parse_args()
    return args


def parse_max_release_date(value: str) -> datetime | None:
    if value.lower() in {"none", "null", ""}:
        return None
    if len(value) == 10:
        value = f"{value} 23:59:59"
    return datetime.fromisoformat(value)


def has_allowed_resolution(
    entry: dict[str, Any],
    min_resolution: float,
    max_resolution: float,
) -> bool:
    exp = entry.get("exp", {})
    resolution = exp.get("resolution")
    return (
        resolution is not None and min_resolution <= float(resolution) <= max_resolution
    )


def has_allowed_release_date(
    entry: dict[str, Any],
    max_release_date: datetime | None,
) -> bool:
    if max_release_date is None:
        return True
    release_date = entry.get("exp", {}).get("release_date")
    if release_date is None:
        return False
    return datetime.fromisoformat(release_date) <= max_release_date


def has_allowed_chain_count(
    entry: dict[str, Any],
    min_chains: int,
    max_chains: int,
) -> bool:
    n_chains = len(entry.get("chains", []))
    return min_chains <= n_chains <= max_chains


def has_allowed_chain_lengths(entry: dict[str, Any], min_residues: int) -> bool:
    chains = entry.get("chains", [])
    return bool(chains) and all(
        int(chain.get("num_residues", 0)) >= min_residues for chain in chains
    )


def has_required_interface(entry: dict[str, Any], require_interface: bool) -> bool:
    if not require_interface:
        return True
    return len(entry.get("interfaces", [])) > 0


def filter_entry(
    entry: dict[str, Any],
    *,
    min_resolution: float,
    max_resolution: float,
    max_release_date: datetime | None,
    min_chains: int,
    max_chains: int,
    min_residues: int,
    require_interface: bool,
) -> bool:
    """Return True if a complex passes manifest-level AF3/RCSB filters.

    The expensive bioassembly filters from AF3 SI 2.5.4, such as all-atom clash
    removal, CA-gap filtering, and closest-20 chain extraction, are applied in
    b1_process.py. This script keeps the lightweight manifest-level guards.
    """
    return (
        has_allowed_resolution(entry, min_resolution, max_resolution)
        and has_allowed_release_date(entry, max_release_date)
        and has_allowed_chain_count(entry, min_chains, max_chains)
        and has_allowed_chain_lengths(entry, min_residues)
        and has_required_interface(entry, require_interface)
    )


def recompute_cluster_sizes(manifest: list[dict[str, Any]]) -> None:
    cluster_size: dict[str, int] = defaultdict(int)
    for entry in manifest:
        for chain in entry.get("chains", []):
            cluster_id = chain.get("cluster_id")
            if cluster_id is not None:
                cluster_size[cluster_id] += 1
        for interface in entry.get("interfaces", []):
            cluster_id = interface.get("cluster_id")
            if cluster_id is not None:
                cluster_size[cluster_id] += 1

    for entry in manifest:
        for chain in entry.get("chains", []):
            cluster_id = chain.get("cluster_id")
            if cluster_id is not None:
                chain["cluster_size"] = cluster_size[cluster_id]
        for interface in entry.get("interfaces", []):
            cluster_id = interface.get("cluster_id")
            if cluster_id is not None:
                interface["cluster_size"] = cluster_size[cluster_id]


def main():
    args = parse_args()
    data_dir = args.data_dir / "rcsb_multimer"
    max_release_date = parse_max_release_date(args.max_release_date)

    print("Loading metadata...")
    manifest_path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        manifest: list[dict[str, Any]] = msgpack.unpackb(f.read(), raw=False)
    print(f"Loaded {len(manifest)} entries from {manifest_path}")

    manifest_filtered = [
        entry
        for entry in manifest
        if filter_entry(
            entry,
            min_resolution=args.min_resolution,
            max_resolution=args.max_resolution,
            max_release_date=max_release_date,
            min_chains=args.min_chains,
            max_chains=args.max_chains,
            min_residues=args.min_residues,
            require_interface=args.require_interface,
        )
    ]

    if args.recompute_cluster_size:
        recompute_cluster_sizes(manifest_filtered)

    print(
        f"Filtered {len(manifest_filtered)} entries with resolution between "
        f"{args.min_resolution} and {args.max_resolution}"
    )
    print(f"  max_release_date: {args.max_release_date}")
    print(f"  chain count: {args.min_chains} to {args.max_chains}")
    print(f"  min residues per chain: {args.min_residues}")
    print(f"  require interface: {args.require_interface}")
    print(f"  recompute cluster size: {args.recompute_cluster_size}")

    out_path = data_dir / f"{args.out_prefix}.msgpack"
    with open(out_path, "wb") as f:
        msgpack.pack(manifest_filtered, f, use_bin_type=True)
    print(f"Saved manifest (msgpack) to {out_path}")


if __name__ == "__main__":
    main()
