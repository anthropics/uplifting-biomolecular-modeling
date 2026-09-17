"""Inputs for ``pred``: the upstream input schema, exactly as upstream defines it.

Upstream defines its own JSON form of a prediction input — ``serialize_structure_prediction_input`` /
``deserialize_structure_prediction_input`` in ``esm/utils/structure/input_builder.py``:

    {"id": "1brs", "seeds": [0, 1],                                   # id / seeds: package keys, optional
     "sequences": [{"type": "protein", "id": "A", "sequence": "...", "msa": null},
                   {"type": "protein", "id": "B", "sequence": "...", "msa": "msas/B.a3m"},      # a path: the package's one extension
                   {"type": "ligand",  "id": "L1", "ccd": ["ATP"]}],
     "covalent_bonds": [...], "pocket": {...}, "distogram_conditioning": [...]}                  # passed through unchanged

One extension: a chain's ``"msa"`` may be a path to an A3M file; the package reads it with upstream's ``MSA.from_a3m(path,
remove_insertions=<settings>, max_sequences=<settings>)`` — the read form is part of the settings (settings.py: the library
defaults keep insertions and read the whole file; ``--remove_insertions`` / ``--max_sequences`` change it) — and attaches the MSA object to the
chain (upstream's inline ``{"sequences": [...]}`` form is passed through untouched). Whether MSAs are used at all is the
variant's decision (``full_msa`` only): for the other variants every chain's MSA is dropped before building (``spi_from_input(item,
use_msa, read_msa)``: the one builder, used by ``pred`` and by the kit server's driver alike).

An input is any upstream input — one chain or many, of any type upstream accepts, in the order given; the package reads no chain roles
and adds no rule of its own (a chain has an MSA iff the input names one). Keys upstream's deserializer does not define are not read by it
(it takes ``sequences``, ``covalent_bonds``, ``pocket``, ``distogram_conditioning``); an extra top-level key is ignored.

An input file is one input (a dict with ``sequences``) or a list of inputs (e.g. the kit's ``tests/w4_public_slice.json``: PDB 1BRS,
single sequence). Nothing here imports the kit; upstream is imported only inside ``spi_from_input``.
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_ID = "pred"                                    # upstream's fold(complex_id=) default
PACKAGE_KEYS = ("id", "seeds", "num_diffusion_samples", "name")   # the package's own top-level keys, removed before upstream's deserializer sees the input


class InputError(ValueError):
    pass


def is_spi(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("sequences"), list)


def normalise(obj: Any, base_dir: str = "") -> List[dict]:
    """Parsed JSON -> a list of upstream-form input dicts with unique ids; MSA paths made absolute against `base_dir`."""
    if is_spi(obj):
        items = [obj]
    elif isinstance(obj, list) and obj and all(is_spi(x) for x in obj):
        items = list(obj)
    else:
        raise InputError("input must be an upstream prediction input ({'sequences': [...]}) or a list of them")
    out: List[dict] = []
    seen = set()
    for i, it in enumerate(items):
        it = dict(it)
        it["id"] = str(it.get("id") or it.get("name") or (DEFAULT_ID if len(items) == 1 else f"item{i:03d}"))
        if it["id"] in seen:
            raise InputError(f"duplicate input id {it['id']!r}")
        seen.add(it["id"])
        seqs = []
        for ch in it["sequences"]:
            ch = dict(ch)
            if not ch.get("type"):
                raise InputError(f"input {it['id']!r}: every chain needs a 'type' (protein|rna|dna|ligand)")
            if ch["type"] != "ligand" and not ch.get("sequence"):
                raise InputError(f"input {it['id']!r}: chain {ch.get('id')!r} has no sequence")
            msa = ch.get("msa")
            if isinstance(msa, str) and base_dir and not os.path.isabs(msa):
                ch["msa"] = os.path.normpath(os.path.join(base_dir, msa))
            seqs.append(ch)
        it["sequences"] = seqs
        out.append(it)
    return out


def load(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return normalise(obj, base_dir=os.path.dirname(os.path.abspath(path)))


def item_seeds(item: dict, default: List[int]) -> List[int]:
    s = item.get("seeds")
    return [int(x) for x in s] if s else list(default)


SAMPLES_KEY = "num_diffusion_samples"                     # an input's own sample count (a package key): read by run_samples / item_samples only


def item_samples(item: dict, default: int) -> int:
    """The diffusion samples ONE input folds at: its own ``num_diffusion_samples`` key when it names one, else `default` (the run's count).
    A run has one count (run_samples refuses a disagreeing input before anything loads), so on the `pred` routes this equals the run's."""
    v = item.get(SAMPLES_KEY)
    n = int(default) if v is None else int(v)
    if n < 1:
        raise InputError(f"input {item.get('id')!r}: num_diffusion_samples={v!r} — a positive integer is required")
    return n


def run_samples(items: List[dict], flag: Optional[int] = None, default: Optional[int] = None) -> Tuple[Optional[int], str]:
    """The ONE sample count a run folds at, and where it came from: ``--num_diffusion_samples`` when given ("flag"), else the inputs' own
    ``num_diffusion_samples`` key — the same on every input ("inputs"), else `default` ("default"; None = the library's own, which the
    caller resolves with the fold settings — nothing of the library is imported here). An input that names a
    count other than the flag's, or two inputs that name different counts, or one input naming a count beside one naming none, is a
    disagreement refused by name (InputError): a run never folds one input at a count its levers, its accounting and its `settings` line
    did not plan for."""
    named = [(it.get("id"), item_samples(it, default)) for it in items if it.get(SAMPLES_KEY) is not None]
    if flag is not None:
        run = int(flag)
        if run < 1:
            raise InputError(f"--num_diffusion_samples {flag}: a positive integer is required")
        other = [(i, n) for i, n in named if n != run]
        if other:
            raise InputError(f"input {other[0][0]!r} names num_diffusion_samples={other[0][1]}, the command line --num_diffusion_samples {run}: "
                             f"one sample count per run — drop the key or the flag ({len(other)} of {len(items)} inputs differ from the flag)")
        return run, "flag"
    if not named:
        return (None if default is None else int(default)), "default"
    counts = sorted({n for _, n in named})
    if len(counts) > 1:
        a = next(i for i, n in named if n == counts[0]); b = next(i for i, n in named if n == counts[-1])
        raise InputError(f"inputs {a!r} and {b!r} name num_diffusion_samples={counts[0]} and {counts[-1]}: one sample count per run — "
                         f"give every input the same count, or pass --num_diffusion_samples")
    unnamed = [it.get("id") for it in items if it.get(SAMPLES_KEY) is None]
    if unnamed:
        raise InputError(f"input {named[0][0]!r} names num_diffusion_samples={counts[0]} and input {unnamed[0]!r} names none: one sample count per run — "
                         f"name it on every input, or pass --num_diffusion_samples")
    return counts[0], "inputs"


# ----------------------------------------------------------------------------------------------------------------- upstream objects
_MSA_CACHE: Dict[Tuple[str, Optional[int], bool], Any] = {}


def read_a3m(path: str, query: str, max_sequences: Optional[int], remove_insertions: bool = False):
    """Upstream ``MSA.from_a3m(path, remove_insertions=..., max_sequences=...)`` with the settings' read form, cached per (path, depth,
    insertions); the query row must equal the chain sequence (the kit driver's own check)."""
    from esm.utils.msa import MSA
    key = (path, max_sequences, bool(remove_insertions))
    if key not in _MSA_CACHE:
        msa = MSA.from_a3m(path=path, remove_insertions=bool(remove_insertions), max_sequences=max_sequences)
        q = msa.sequences[0].upper().replace("-", "")
        if q != query:
            raise InputError(f"MSA query != chain sequence for {path}")
        _MSA_CACHE[key] = msa
    return _MSA_CACHE[key]


def spi_from_input(item: dict, use_msa: bool, read_msa: Callable[[str, str], Any]):
    """An input dict -> upstream ``StructurePredictionInput`` (upstream's own deserializer; the chains as given). ``use_msa`` False drops
    every chain's MSA first; a chain's A3M path is read with ``read_msa(path, chain_sequence)`` and attached afterwards. The ONE builder:
    ``build_spi`` (pred: the settings' read form) calls it."""
    from esm.utils.structure.input_builder import deserialize_structure_prediction_input
    data = {k: v for k, v in item.items() if k not in PACKAGE_KEYS}
    paths: List[Optional[str]] = []
    seqs = []
    for ch in data["sequences"]:
        ch = dict(ch)
        msa = ch.get("msa")
        if not use_msa:
            msa = None
        if isinstance(msa, str):
            paths.append(msa); ch["msa"] = None
        else:
            paths.append(None); ch["msa"] = msa
        seqs.append(ch)
    data["sequences"] = seqs
    spi = deserialize_structure_prediction_input(data)
    for seq_input, p in zip(spi.sequences, paths):
        if p is not None:
            seq_input.msa = read_msa(p, seq_input.sequence)
    return spi


def build_spi(item: dict, use_msa: bool, msa_read_depth: Optional[int] = None, remove_insertions: bool = False):
    """An input dict -> upstream ``StructurePredictionInput`` with the settings' A3M read form (``msa_read_depth`` = ``max_sequences``,
    ``remove_insertions``)."""
    return spi_from_input(item, use_msa, lambda path, query: read_a3m(path, query, msa_read_depth, remove_insertions))


def msa_depths(spi) -> List[int]:
    return [int(getattr(getattr(s, "msa", None), "depth", 0) or 0) for s in spi.sequences]
