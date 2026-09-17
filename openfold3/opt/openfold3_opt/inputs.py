"""The call's query file as the package reads it: upstream's query JSON (``{"queries": {...}}``, read as is) and the polymer token counts the
size gates take from it — per query (polymer_tokens_each: the offload port's item gate judges every item on its own count, cli.item_gate_groups) and
the largest query's (polymer_tokens: modes.graphs_gate / conf_gate, the protein/RNA/DNA residues x copies). No conversion, no schema of the package's
own: ``pred`` takes upstream's ``--query-json`` exactly as ``run_openfold predict`` does; a subset document (subset) is the same document with fewer
``queries`` entries."""
import json
from typing import Dict, Iterable, Tuple

POLYMERS = ("protein", "rna", "dna")


class InputError(ValueError):
    pass


def is_query_set(obj) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("queries"), dict)


def load(path: str) -> dict:
    """Upstream's query JSON, unchanged; a document that is not a query set is refused by name (InputError)."""
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if is_query_set(obj):
        return obj
    raise InputError(f"{path}: not a query set (upstream's {{\"queries\": {{...}}}})")


def query_polymer_tokens(query: dict) -> Tuple[int, int]:
    """(n_tokens, n_other) of ONE query: n_tokens = the residues of its protein/RNA/DNA chains × their copies (len(chain_ids)), one token each — a
    lower bound of the model's token count (ligand atoms are tokens too and are not counted here); n_other = its chains of any other molecule type
    (ligands), counted as entries."""
    n_tok = n_other = 0
    for ch in query.get("chains") or []:
        ids = ch.get("chain_ids")
        copies = len(ids) if isinstance(ids, (list, tuple)) else 1
        if str(ch.get("molecule_type", "")).lower() in POLYMERS and isinstance(ch.get("sequence"), str):
            n_tok += len(ch["sequence"]) * max(copies, 1)
        else:
            n_other += max(copies, 1)
    return n_tok, n_other


def polymer_tokens_each(query_set: dict) -> Dict[str, Tuple[int, int]]:
    """name -> (n_tokens, n_other) for every query of the set, in the document's order (query_polymer_tokens). The offload port's item gate reads it
    per item (cli.item_gate_groups -> modes.of3o_item_groups)."""
    return {str(name): query_polymer_tokens(q) for name, q in (query_set.get("queries") or {}).items()}


def polymer_tokens(query_set: dict) -> Tuple[int, int]:
    """(n_tokens, n_other) of the LARGEST query in the set (query_polymer_tokens per query; the largest pair). The CUDA-graph size gate reads n_tokens
    (modes.graphs_gate); a `pred` pass reads its own query file, so under the per-item port gate this is the pass's largest member."""
    best = (0, 0)
    for nt in polymer_tokens_each(query_set).values():
        best = max(best, nt)
    return best


def subset(query_set: dict, names: Iterable[str]) -> dict:
    """The same document with ``queries`` narrowed to `names` (kept in the document's order; every other top-level key unchanged). A name the set does
    not hold is refused by name (InputError)."""
    want = [str(n) for n in names]
    queries = query_set.get("queries") or {}
    missing = [n for n in want if n not in queries]
    if missing:
        raise InputError(f"subset: no such quer{'y' if len(missing) == 1 else 'ies'} in the set: {', '.join(missing)}")
    keep = set(want)
    return {**query_set, "queries": {k: v for k, v in queries.items() if str(k) in keep}}


def dump(query_set: dict, path: str) -> None:
    """Write a query document (json, utf-8) — the per-pass subset files of cli.run_item_groups."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(query_set, fh, indent=1)
