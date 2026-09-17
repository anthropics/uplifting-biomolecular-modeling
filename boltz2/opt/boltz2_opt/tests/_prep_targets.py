"""Targets the zygote tests run in its children (prep.Zygote.run takes ``"module:function"``): importable, side effects visible in their output."""
import hashlib
import os
import sys

CALLS = 0                                     # per-process state: a fresh child of the zygote always sees the zygote's value, never a sibling's


def bump(tag: str, fail: bool = False, die: int = 0, payload=None) -> None:
    global CALLS
    CALLS += 1
    print(f"out tag={tag} pid={os.getpid()} calls={CALLS} payload={payload!r}")
    print(f"err tag={tag}", file=sys.stderr)
    if die:
        os._exit(die)                         # a child that dies without reporting
    if fail:
        raise ValueError(f"boom {tag}")


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"             # the SMILES of entity_ligand_smiles_aspirin_on_1l2y and extra_boltz2_affinity_1l2y (cofold_identity_v1)


def conformer_digest(smiles: str = ASPIRIN) -> str:
    """sha256 of the conformer coordinates boltz gives a SMILES ligand (schema.py: AddHs, then compute_3d_conformer = ETKDGv3 with no
    randomSeed + UFF), computed the way this PROCESS's RDKit RNG stands: boltz's own function when boltz is importable, else the same three
    RDKit calls."""
    from rdkit.Chem import AllChem
    mol = AllChem.AddHs(AllChem.MolFromSmiles(smiles))
    try:
        from boltz.data.parse.schema import compute_3d_conformer
        compute_3d_conformer(mol)
    except ImportError:
        options = AllChem.ETKDGv3(); options.clearConfs = False
        conf_id = AllChem.EmbedMolecule(mol, options)
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
    return hashlib.sha256(mol.GetConformer().GetPositions().tobytes()).hexdigest()


def print_conformer_digest(smiles: str = ASPIRIN) -> None:
    print(conformer_digest(smiles))


def read_relative(path: str) -> None:
    """Opens ``path`` RELATIVE to the child's cwd and prints it with the cwd — a parse child inherits the worker's directory (worker_launch.enter_invoking_dir)."""
    print(f"cwd={os.getcwd()} content={open(path).read().strip()}")

