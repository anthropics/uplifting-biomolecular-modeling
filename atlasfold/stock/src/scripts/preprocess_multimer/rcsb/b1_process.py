"""Preprocess RCSB mmCIF files into multimer complexes."""

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import multiprocessing
import os
import pathlib
from datetime import datetime

import gemmi
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tqdm import tqdm

from atlasfold.common import metadata, protein
from atlasfold.data import cif_factory
from atlasfold.train.multimer.dataset import MultimerDataPipeline

# Error handling
SUCCESS = 0
FAILED = 1
DATE_FILTERED = 2
RESOLUTION_FILTERED = 3
NO_PROTEIN_FILTERED = 4
CHAIN_COUNT_FILTERED = 5
NO_VALID_CHAINS_FILTERED = 6
TOKEN_COUNT_FILTERED = 7

MIN_LENGTH: int = 4
MAX_EXTRACTED_CHAINS: int = 20
INTERFACE_CA_DISTANCE: float = 15.0
CONTACT_DISTANCE: float = 5.0
CLASH_DISTANCE: float = 1.7
MAX_CLASH_RATIO: float = 0.3

logger = logging.getLogger(__name__)

GLOBAL_CLUSTER_MAPPING: dict[str, str] = {}


@dataclasses.dataclass(frozen=True)
class DataFilter:
    """RCSB multimer split filters."""

    output_name: str
    date_start: datetime
    date_end: datetime
    max_resolution: float
    min_chains: int
    max_input_chains: int
    max_output_chains: int = MAX_EXTRACTED_CHAINS
    max_residues: int | None = None
    require_cluster_mapping: bool = True


AF3_SPLITS: dict[str, DataFilter] = {
    "train": DataFilter(
        output_name="rcsb_multimer",
        date_start=datetime.min,
        date_end=datetime.fromisoformat("2021-09-30 23:59:59"),
        max_resolution=9.0,
        min_chains=1,
        max_input_chains=300,
        require_cluster_mapping=True,
    ),
    "val": DataFilter(
        output_name="rcsb_multimer_val",
        date_start=datetime.fromisoformat("2021-10-01 00:00:00"),
        date_end=datetime.fromisoformat("2023-01-12 23:59:59"),
        max_resolution=4.5,
        min_chains=2,
        max_input_chains=1000,
        max_output_chains=20,
        max_residues=1536,
        require_cluster_mapping=False,
    ),
}


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
        "--split",
        type=str,
        default="train",
        choices=sorted(AF3_SPLITS),
        help="AF3 split to process. Validation outputs go to rcsb_multimer_val.",
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
def expand_first_assembly(raw_struct: gemmi.Structure) -> None:
    """Expand the first biological assembly in-place when assembly data exists."""
    if len(raw_struct.assemblies) == 0:
        return
    how = gemmi.HowToNameCopiedChain.AddNumber
    try:
        raw_struct.transform_to_assembly(raw_struct.assemblies[0].name, how=how)
    except Exception as e:
        logger.warning(f"Failed to expand first assembly: {e}")


def validate_sequence(prot: protein.Protein) -> bool:
    """Return True if the chain sequence is not dominated by one residue type."""
    sequence = prot.sequence
    if len(sequence) < MIN_LENGTH:
        logger.debug(f"{prot.name} marked invalid due to short sequence.")
        return False
    aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
    if any(count / len(sequence) > 0.8 for count in aa_counts.values()):
        logger.debug(
            f"{prot.name} marked invalid due to "
            f"overrepresentation of a single amino acid."
        )
        return False
    return True


def validate_geometry(prot: protein.Protein) -> bool:
    """Return True if a protein chain has enough resolved CA atoms and no big jumps."""
    ca_coords = prot.coordinates[:, 1, :]
    is_resolved = np.isfinite(ca_coords).all(axis=-1)
    n_resolved = int(np.sum(is_resolved))
    if n_resolved < MIN_LENGTH:
        logger.debug(f"{prot.name} marked invalid due to insufficient resolved CA atoms.")
        return False

    adjacent_resolved = is_resolved[:-1] & is_resolved[1:]
    dists = np.linalg.norm(ca_coords[:-1] - ca_coords[1:], axis=-1)
    if np.any(dists[adjacent_resolved] > 10.0):
        logger.debug(f"{prot.name} marked invalid due to CA trace discontinuity.")
        return False
    return True


def has_protein(raw_struct: gemmi.Structure) -> bool:
    """Return True if the structure has at least one protein entity."""
    for entity in raw_struct.entities:
        if (
            entity.entity_type == gemmi.EntityType.Polymer
            and entity.polymer_type == gemmi.PolymerType.PeptideL
            and len(entity.full_sequence) >= MIN_LENGTH
        ):
            return True
    return False


def detect_interfaces(
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
) -> list[metadata.InterfaceMetadata]:
    """Detect protein-protein interfaces using CA-CA distances."""
    interfaces: list[metadata.InterfaceMetadata] = []
    ca_coords = [c.coordinates[:, 1, :] for c in chains]
    ca_masks = [np.isfinite(coords).all(axis=-1) for coords in ca_coords]

    for i in range(len(chains)):
        if not np.any(ca_masks[i]):
            continue
        for j in range(i + 1, len(chains)):
            if not np.any(ca_masks[j]):
                continue
            d = cdist(ca_coords[i][ca_masks[i]], ca_coords[j][ca_masks[j]])
            if not np.any(d < INTERFACE_CA_DISTANCE):
                continue

            ci = chain_metadatas[i].cluster_id
            cj = chain_metadatas[j].cluster_id
            if ci is None or cj is None:
                cluster_id = None
            else:
                cluster_id = "__".join(sorted((ci, cj)))

            interfaces.append(
                metadata.InterfaceMetadata(
                    chain_ids=(i, j),
                    cluster_id=cluster_id,
                )
            )
    return interfaces


def get_resolved_atom_coordinates(chain: protein.Protein) -> np.ndarray:
    """Return finite atom coordinates for a chain."""
    coords = chain.coordinates.reshape(-1, 3)
    return coords[np.isfinite(coords).all(axis=-1)]


def filter_clashing_chains(
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
) -> tuple[list[protein.Protein], list[metadata.Metadata]]:
    """Remove chains with severe all-atom clashes.

    This mirrors the AF3/KFold criterion: chain pairs in contact are checked for
    atoms closer than 1.7A, and chains with more than 30% clashing atoms are
    removed.
    """
    invalid_indices: set[int] = set()
    atom_coords = [get_resolved_atom_coordinates(c) for c in chains]
    boxes: list[tuple[np.ndarray, np.ndarray] | None] = []
    trees: list[cKDTree | None] = []
    for coords in atom_coords:
        if len(coords) == 0:
            boxes.append(None)
            trees.append(None)
        else:
            boxes.append((coords.min(axis=0), coords.max(axis=0)))
            trees.append(cKDTree(coords))

    for i in range(len(chains)):
        if i in invalid_indices or trees[i] is None or boxes[i] is None:
            continue
        for j in range(i + 1, len(chains)):
            if j in invalid_indices or trees[j] is None or boxes[j] is None:
                continue

            min_i, max_i = boxes[i]
            min_j, max_j = boxes[j]
            if np.any(min_i - max_j > CONTACT_DISTANCE) or np.any(
                min_j - max_i > CONTACT_DISTANCE
            ):
                continue

            tree_i = trees[i]
            tree_j = trees[j]
            assert tree_i is not None and tree_j is not None
            if tree_i.count_neighbors(tree_j, r=CONTACT_DISTANCE) == 0:
                continue
            if tree_i.count_neighbors(tree_j, r=CLASH_DISTANCE) == 0:
                continue

            clash_indices_i = tree_i.query_ball_tree(tree_j, r=CLASH_DISTANCE)
            clash_indices_j = tree_j.query_ball_tree(tree_i, r=CLASH_DISTANCE)
            n_clash_atoms_i = sum(1 for neighbors in clash_indices_i if neighbors)
            n_clash_atoms_j = sum(1 for neighbors in clash_indices_j if neighbors)
            n_atoms_i = len(atom_coords[i])
            n_atoms_j = len(atom_coords[j])
            clash_ratio_i = n_clash_atoms_i / n_atoms_i
            clash_ratio_j = n_clash_atoms_j / n_atoms_j
            is_clash_i = clash_ratio_i > MAX_CLASH_RATIO
            is_clash_j = clash_ratio_j > MAX_CLASH_RATIO

            if is_clash_i and is_clash_j:
                if clash_ratio_i > clash_ratio_j:
                    remove_i = i
                elif clash_ratio_i < clash_ratio_j:
                    remove_i = j
                elif n_atoms_i > n_atoms_j:
                    remove_i = j
                elif n_atoms_i < n_atoms_j:
                    remove_i = i
                else:
                    asym_i = chain_metadatas[i].asym_id or i
                    asym_j = chain_metadatas[j].asym_id or j
                    remove_i = i if asym_i > asym_j else j
                invalid_indices.add(remove_i)
                logger.debug(
                    f"{chain_metadatas[remove_i].id} marked invalid due to severe "
                    f"clash between chains {chain_metadatas[i].asym_id} and "
                    f"{chain_metadatas[j].asym_id}."
                )
                if remove_i == i:
                    break
            elif is_clash_i:
                invalid_indices.add(i)
                logger.debug(
                    f"{chain_metadatas[i].id} marked invalid due to clash "
                    f"({clash_ratio_i:.2%} atoms clashed with chain "
                    f"{chain_metadatas[j].asym_id})."
                )
                break
            elif is_clash_j:
                invalid_indices.add(j)
                logger.debug(
                    f"{chain_metadatas[j].id} marked invalid due to clash "
                    f"({clash_ratio_j:.2%} atoms clashed with chain "
                    f"{chain_metadatas[i].asym_id})."
                )

    filtered_chains = [c for i, c in enumerate(chains) if i not in invalid_indices]
    filtered_metadatas = [
        m for i, m in enumerate(chain_metadatas) if i not in invalid_indices
    ]
    return filtered_chains, filtered_metadatas


def reindex_chain_metadata(chain_metadatas: list[metadata.Metadata]) -> None:
    """Reassign contiguous asym/sym ids after filtering or subcomplex extraction."""
    sym_id_by_entity: dict[int, int] = {}
    for asym_id, m in enumerate(chain_metadatas, 1):
        m.asym_id = asym_id
        assert m.entity_id is not None
        sym_id_by_entity[m.entity_id] = sym_id_by_entity.get(m.entity_id, 0) + 1
        m.sym_id = sym_id_by_entity[m.entity_id]


def get_resolved_ca_coordinates(chain: protein.Protein) -> np.ndarray:
    """Return finite CA coordinates for a chain."""
    ca_coords = chain.coordinates[:, 1, :]
    return ca_coords[np.isfinite(ca_coords).all(axis=-1)]


def select_closest_chains(
    name: str,
    chains: list[protein.Protein],
    chain_metadatas: list[metadata.Metadata],
    interfaces: list[metadata.InterfaceMetadata],
    max_chains: int = MAX_EXTRACTED_CHAINS,
) -> tuple[list[protein.Protein], list[metadata.Metadata]]:
    """Select up to max_chains around a sampled interface/contact seed.

    This follows the AF3 SI 2.5.4 subcomplex extraction idea: for assemblies with
    many chains, sample an interface token and keep the chains closest to it.
    """
    if len(chains) <= max_chains:
        return chains, chain_metadatas

    seed = int.from_bytes(hashlib.sha1(name.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    ca_coords = [get_resolved_ca_coordinates(c) for c in chains]
    valid_chain_indices = [i for i, coords in enumerate(ca_coords) if len(coords) > 0]
    if len(valid_chain_indices) <= max_chains:
        selected_indices = set(valid_chain_indices)
    else:
        seed_coords = None
        if interfaces:
            iface = interfaces[rng.integers(len(interfaces))]
            chain_i, chain_j = iface.chain_ids
            coords_i = ca_coords[chain_i]
            coords_j = ca_coords[chain_j]
            if len(coords_i) > 0 and len(coords_j) > 0:
                d = cdist(coords_i, coords_j)
                is_contact = d < INTERFACE_CA_DISTANCE
                if np.any(is_contact):
                    if rng.random() < 0.5:
                        contact_indices = np.where(np.any(is_contact, axis=1))[0]
                        seed_coords = coords_i[rng.choice(contact_indices)]
                    else:
                        contact_indices = np.where(np.any(is_contact, axis=0))[0]
                        seed_coords = coords_j[rng.choice(contact_indices)]
                elif rng.random() < 0.5:
                    seed_coords = coords_i[rng.integers(len(coords_i))]
                else:
                    seed_coords = coords_j[rng.integers(len(coords_j))]

        if seed_coords is None:
            chain_i = valid_chain_indices[rng.integers(len(valid_chain_indices))]
            coords_i = ca_coords[chain_i]
            seed_coords = coords_i[rng.integers(len(coords_i))]

        chain_dists = []
        for i in valid_chain_indices:
            d = np.linalg.norm(ca_coords[i] - seed_coords[None, :], axis=-1)
            chain_dists.append((i, float(np.min(d))))
        chain_dists.sort(key=lambda x: x[1])
        selected_indices = {i for i, _ in chain_dists[:max_chains]}

    selected_chains = [c for i, c in enumerate(chains) if i in selected_indices]
    selected_metadatas = [
        m for i, m in enumerate(chain_metadatas) if i in selected_indices
    ]
    return selected_chains, selected_metadatas


def parse_cif(
    name: str,
    cif_path: pathlib.Path,
    out_dir: pathlib.Path,
    data_filter: DataFilter,
) -> tuple[int, int, int]:
    """Parse one CIF file and save the processed complex."""
    global GLOBAL_CLUSTER_MAPPING
    if data_filter.require_cluster_mapping:
        assert GLOBAL_CLUSTER_MAPPING, "Cluster mapping is not initialized in worker."

    npz_path = out_dir / f"{name}.npz"
    json_path = out_dir / f"{name}.json"
    if npz_path.exists() and json_path.exists():
        return SUCCESS, 1, 0

    # Read CIF file.
    block: gemmi.cif.Block = gemmi.cif.read(str(cif_path))[0]
    exp_record = cif_factory.read_experiment_record(block)

    # Filter by date.
    release_date = datetime.fromisoformat(exp_record.release_date)
    if not data_filter.date_start <= release_date <= data_filter.date_end:
        return DATE_FILTERED, 0, 0

    # Filter by resolution.
    resolution = exp_record.resolution
    if resolution is not None and resolution > data_filter.max_resolution:
        return RESOLUTION_FILTERED, 0, 0

    # Prepare gemmi structure.
    raw_struct: gemmi.Structure = gemmi.make_structure_from_block(block)
    expand_first_assembly(raw_struct)
    cif_factory.clean_up_gemmi_structure(raw_struct)

    if not has_protein(raw_struct):
        return NO_PROTEIN_FILTERED, 0, 0

    parsed_chains: list[tuple[protein.Protein, metadata.Metadata]] = []
    sym_id_by_entity: dict[int, int] = {}
    for asym_id, (c, ids) in enumerate(cif_factory.get_protein_chains(raw_struct), 1):
        label_id = "".join(c for c in ids["label_id"] if c.isalpha())
        auth_id = "".join(c for c in ids["auth_id"] if c.isalpha())
        entity_id = ids["entity_id"]
        c.name = f"{name}_{label_id}"

        cluster_id = None
        if data_filter.require_cluster_mapping:
            key = f"{name}_{entity_id}".upper()
            cluster_id = GLOBAL_CLUSTER_MAPPING.get(key)
            if cluster_id is None:
                logger.warning(
                    f"Cluster ID not found for entity_id {entity_id} in {name}."
                )
                continue

        sym_id_by_entity[entity_id] = sym_id_by_entity.get(entity_id, 0) + 1
        m = metadata.Metadata(
            id=f"{name}_{label_id}",
            label_asym_id=label_id,
            auth_asym_id=auth_id,
            entity_id=entity_id,
            asym_id=asym_id,
            sym_id=sym_id_by_entity[entity_id],
            num_residues=len(c),
            cluster_id=cluster_id,
        )
        parsed_chains.append((c, m))

    if not (data_filter.min_chains <= len(parsed_chains) <= data_filter.max_input_chains):
        return CHAIN_COUNT_FILTERED, 0, 0

    # Check sequence and geometry.
    valid_chains = [
        (c, m)
        for (c, m) in parsed_chains
        if validate_sequence(c) and validate_geometry(c)
    ]
    if not valid_chains:
        return NO_VALID_CHAINS_FILTERED, 0, 0
    if not (data_filter.min_chains <= len(valid_chains) <= data_filter.max_input_chains):
        return CHAIN_COUNT_FILTERED, 0, 0

    chains = [c for c, _ in valid_chains]
    chain_metadatas = [m for _, m in valid_chains]
    reindex_chain_metadata(chain_metadatas)
    chains, chain_metadatas = filter_clashing_chains(chains, chain_metadatas)
    if not chains:
        return NO_VALID_CHAINS_FILTERED, 0, 0
    if not (data_filter.min_chains <= len(chains) <= data_filter.max_input_chains):
        return CHAIN_COUNT_FILTERED, 0, 0
    reindex_chain_metadata(chain_metadatas)
    interfaces = detect_interfaces(chains, chain_metadatas)
    chains, chain_metadatas = select_closest_chains(
        name,
        chains,
        chain_metadatas,
        interfaces,
        max_chains=data_filter.max_output_chains,
    )
    reindex_chain_metadata(chain_metadatas)
    interfaces = detect_interfaces(chains, chain_metadatas)
    if not (data_filter.min_chains <= len(chains) <= data_filter.max_output_chains):
        return CHAIN_COUNT_FILTERED, 0, 0
    if (
        data_filter.max_residues is not None
        and sum(len(c) for c in chains) > data_filter.max_residues
    ):
        return TOKEN_COUNT_FILTERED, 0, 0

    compl = protein.ProteinMultimer(
        name=name,
        chains=chains,
        entity_ids=[m.entity_id for m in chain_metadatas if m.entity_id is not None],
        asym_ids=[m.asym_id for m in chain_metadatas if m.asym_id is not None],
        sym_ids=[m.sym_id for m in chain_metadatas if m.sym_id is not None],
    )
    complex_metadata = metadata.MultimerMetadata(
        id=name,
        chains=chain_metadatas,
        interfaces=interfaces,
        exp=exp_record,
    )

    MultimerDataPipeline.save(compl, npz_path)
    with open(json_path, "w") as f:
        json.dump(complex_metadata.to_dict(), f)

    return SUCCESS, 1, len(chains)


def worker_fn(
    cif_path: pathlib.Path,
    output_dir: pathlib.Path,
    data_filter: DataFilter,
):
    """Worker function to process a single CIF file."""
    pdb_id = cif_path.name.split(".")[0].lower()
    out_dir = output_dir / pdb_id[1:3] / pdb_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return parse_cif(pdb_id, cif_path, out_dir, data_filter)
    except Exception as e:
        logger.error(f"Failed to process ({pdb_id}): {e}")
        return FAILED, 0, 0


def worker_init(cluster_path: pathlib.Path | None):
    """Initialize worker process with global entity-to-cluster mapping."""
    global GLOBAL_CLUSTER_MAPPING
    if cluster_path is None:
        GLOBAL_CLUSTER_MAPPING = {}
        return

    cluster_mapping = {}
    with open(cluster_path) as f:
        next(f)  # skip header
        for line in f:
            entity_id, cluster_id, _ = line.strip().split(",", 2)
            cluster_mapping[entity_id.upper()] = cluster_id.upper()
    GLOBAL_CLUSTER_MAPPING = cluster_mapping


def main():
    """Main function to process RCSB mmCIF files."""
    args = parse_args()
    cif_dir: pathlib.Path = args.cif_dir
    data_filter = AF3_SPLITS[args.split]
    data_dir: pathlib.Path = args.data_dir / data_filter.output_name

    cluster_mapping_path: pathlib.Path | None = None
    if data_filter.require_cluster_mapping:
        cluster_mapping_path = data_dir / "rcsb_clusters.csv"
        assert cluster_mapping_path.exists(), (
            f"Cluster mapping file not found: {cluster_mapping_path}"
        )

    print(f"Applying AF3 {args.split} split parameters:")
    print(data_filter)

    out_dir: pathlib.Path = data_dir / "npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    worker_wrapped = functools.partial(
        worker_fn,
        output_dir=out_dir,
        data_filter=data_filter,
    )

    cif_paths = sorted(cif_dir.rglob("*.cif.gz"))
    print(f"Found {len(cif_paths)} mmCIF files to process.")

    flags: list[int] = []
    n_complexes = 0
    n_chains = 0
    with multiprocessing.Pool(
        args.num_workers,
        initializer=worker_init,
        initargs=(cluster_mapping_path,),
    ) as pool:
        pbar = tqdm(
            pool.imap_unordered(worker_wrapped, cif_paths, chunksize=10),
            total=len(cif_paths),
            desc="Processing RCSB mmCIF files",
        )
        for flag, n_success, n_success_chains in pbar:
            n_complexes += n_success
            n_chains += n_success_chains
            flags.append(flag)
            pbar.set_postfix({"Multimeres": n_complexes, "Chains": n_chains})
    print("Processing completed.")

    print("Processing statistics:")
    print(f"  Total files processed: {len(flags)}")
    print(f"  Successfully processed: {flags.count(SUCCESS)}")
    print(f"  Failed to process: {flags.count(FAILED)}")
    print(f"  Date filtered: {flags.count(DATE_FILTERED)}")
    print(f"  Resolution filtered: {flags.count(RESOLUTION_FILTERED)}")
    print(f"  No protein filtered: {flags.count(NO_PROTEIN_FILTERED)}")
    print(f"  Chain count filtered: {flags.count(CHAIN_COUNT_FILTERED)}")
    print(f"  No valid chains filtered: {flags.count(NO_VALID_CHAINS_FILTERED)}")
    print(f"  Token count filtered: {flags.count(TOKEN_COUNT_FILTERED)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
