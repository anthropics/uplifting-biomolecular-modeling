"""Readers of the fork's fold-input JSON (dialect ``alphafold3``): the seed count a pass will predict, the templates it declares, and the
token estimate the memory mode sizes its program by (token_estimate — the one reader; pred, check and warm resolve through it)."""
from __future__ import annotations

import json
import os

DIALECT = "alphafold3"
POLYMER_KINDS = ("protein", "rna", "dna")


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seeds_in(path_or_dir: str, num_seeds=None) -> int:
    """Total seeds over the JSON file(s) — the expected prediction count of a pass, per diffusion sample: each file's own modelSeeds, or,
    when the run states ``--num_seeds N`` (outputs.num_seeds), N per file — the stock script expands every fold input to N seeds
    (run_alphafold.py: fold_input.with_multiple_seeds(N))."""
    files = [os.path.join(path_or_dir, f) for f in sorted(os.listdir(path_or_dir)) if f.endswith(".json")] if os.path.isdir(path_or_dir) else [path_or_dir]
    if num_seeds is not None:
        return int(num_seeds) * len(files)
    return sum(len(load(p).get("modelSeeds", [])) for p in files)


def templates_declared(path_or_dir: str) -> int:
    """Templates declared over the JSON file(s): the count of entries in every polymer chain's ``templates`` list (absent / null = 0). A run whose
    input declares templates is a templated run: the model process's template census guard (inprocess/templates.py) accounts for every one."""
    files = [os.path.join(path_or_dir, f) for f in sorted(os.listdir(path_or_dir)) if f.endswith(".json")] if os.path.isdir(path_or_dir) else [path_or_dir]
    n = 0
    for p in files:
        for ent in load(p).get("sequences", []) or []:
            for kind in POLYMER_KINDS:
                body = ent.get(kind) if isinstance(ent, dict) else None
                if isinstance(body, dict):
                    n += len(body.get("templates") or [])
    return n


LIGAND_CCD_TOKENS = 100                                                 # tokens counted per CCD code whose atom count the wrapper cannot read up front (conservative: an over-count moves an
                                                                        # input toward the memory mode's reach region, never the other way)
_SMILES_ATOM_RX = None


def smiles_heavy_atoms(smiles: str) -> int:
    """Heavy atoms in a SMILES string (one token each in the fork's tokenisation of a ligand): bracket atoms plus the organic-subset symbols
    outside brackets; hydrogens are not tokens. A count, not a parser — malformed SMILES over-counts at worst."""
    import re
    global _SMILES_ATOM_RX
    if _SMILES_ATOM_RX is None:
        _SMILES_ATOM_RX = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOPSFIbcnops]")
    n = 0
    for tok in _SMILES_ATOM_RX.findall(smiles or ""):
        if tok.startswith("["):
            n += 0 if re.fullmatch(r"\[\d*H\d*[+-]?\d*\]", tok) else 1
        else:
            n += 1
    return n


def token_estimate(path_or_dir: str) -> dict:
    """The token count the model will see, estimated up front from the fold-input JSON(s) (the memory mode's region rule reads it:
    modes.big_region): per polymer chain len(sequence) per copy (``id`` a string = 1 copy, a list = that many); per ligand its SMILES heavy
    atoms or LIGAND_CCD_TOKENS per CCD code, per copy. ``{'n_est': the LARGEST input's estimate | None, 'files': n, 'per_file': {name: n},
    'unreadable': [name: why]}`` — an input that is not a readable alphafold3-dialect fold input makes n_est None (the rule then names it)."""
    files = [os.path.join(path_or_dir, f) for f in sorted(os.listdir(path_or_dir)) if f.endswith(".json")] if os.path.isdir(path_or_dir) else [path_or_dir]
    per, bad = {}, []
    for p in files:
        name = os.path.basename(p)
        try:
            fi = load(p)
            seqs = fi.get("sequences") if isinstance(fi, dict) else None
            if not isinstance(seqs, list) or not seqs:
                raise ValueError("no 'sequences' list (not an alphafold3-dialect fold input)")
            n = 0
            for ent in seqs:
                if not isinstance(ent, dict) or len(ent) != 1:
                    raise ValueError(f"entity {str(ent)[:60]!r} is not a one-key mapping")
                kind, body = next(iter(ent.items()))
                ids = body.get("id") if isinstance(body, dict) else None
                copies = len(ids) if isinstance(ids, list) else 1
                if kind in POLYMER_KINDS:
                    n += len(str(body.get("sequence") or "")) * copies
                elif kind == "ligand":
                    if body.get("smiles"):
                        n += smiles_heavy_atoms(str(body["smiles"])) * copies
                    else:
                        n += LIGAND_CCD_TOKENS * max(1, len(body.get("ccdCodes") or [])) * copies
                else:
                    raise ValueError(f"entity kind {kind!r} unknown")
            per[name] = n
        except (OSError, ValueError, AttributeError, StopIteration, TypeError) as e:
            bad.append(f"{name}: {e}")
    n_est = None if (bad or not per) else max(per.values())
    return {"n_est": n_est, "files": len(files), "per_file": per, "unreadable": bad}

