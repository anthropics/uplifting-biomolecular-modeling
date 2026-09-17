"""Outputs: the index of what a run produced, hashed outside the producing process.

CLI route (`opendde pred`): `<out>/<name>/seed_<S>/predictions/<name>_sample_<k>.cif`, `<name>_summary_confidence_sample_<k>.json`,
`<name>_full_data_sample_<k>.json` per (item, seed, sample) (runner/dumper.py:125-135 `_get_dump_dir`, :137-199: the files are written into a
staging directory `seed_<S>/.predictions-staging-*` renamed onto `predictions/` atomically, so a `predictions/` directory is always a complete
seed; `<out>/ERR/` holds upstream's per-invocation error reports and is removed when empty, runner/inference.py:172-190,1299-1311,1879-1886;
a sample that failed makes the stock exit non-zero, :1893-1897).

Every file is indexed by its byte sha256 (`sha256`) and its layout key with the item name removed from the path (`key`), so the same
item run under two arms lines up; `sha256_name_normalised` hashes text outputs with the item name replaced by `NAME` (the
normalisation rule is labelled here and in the manifest).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

TEXT_SUFFIXES = (".json", ".cif", ".pdb", ".csv", ".jsonl", ".txt")
PACKAGE_FILES = ("opt_manifest.json",)   # the package's own file beside the outputs: never indexed


def sha256_bytes(b: bytes) -> str:
    """Digest of in-memory bytes (the index hashes each file's bytes AND its name-normalised bytes: not a file digest, so not opt_core.gates.sha256_file)."""
    return hashlib.sha256(b).hexdigest()


def _index_file(path: str, key: str, name: str | None) -> dict:
    with open(path, "rb") as fh:
        b = fh.read()
    row = {"path": path, "key": key, "bytes": len(b), "sha256": sha256_bytes(b)}
    if name and path.endswith(TEXT_SUFFIXES):
        row["sha256_name_normalised"] = sha256_bytes(b.replace(name.encode(), b"NAME"))
    return row


def index_cli(out_dir: str) -> dict:
    """Index `<out>/<name>/seed_<S>/predictions/*` (and anything else upstream wrote under <out>)."""
    rows = []
    out_dir = os.path.abspath(out_dir)
    for item in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        ip = os.path.join(out_dir, item)
        if not os.path.isdir(ip):
            if item not in PACKAGE_FILES:
                rows.append(_index_file(ip, item, None))
            continue
        for seed_dir in sorted(os.listdir(ip)):
            sp = os.path.join(ip, seed_dir)
            if not os.path.isdir(sp):
                rows.append(_index_file(sp, f"{item}/{seed_dir}".replace(item, "ITEM", 1), item))
                continue
            for dp, _dn, fn in os.walk(sp):
                for f in sorted(fn):
                    p = os.path.join(dp, f)
                    rel = os.path.relpath(p, ip)
                    key = "ITEM/" + rel.replace(item, "NAME")
                    rows.append(_index_file(p, key, item))
    return {"route": "cli", "out_dir": out_dir, "n_files": len(rows), "files": rows,
            "name_normalisation": "item name replaced by NAME in text outputs (labelled; the rule as shipped is not stated)"}


def expected_structures(stock_args: list, jobs: list) -> dict:
    """The structure files a `pred` run owes: every query item x every seed x every sample (`--seeds a,b` / `-s` / the item's own `modelSeeds`
    when the arguments carry none, else one seed; `--sample` / `-e`; the LAST occurrence of a flag wins, as click reads them)."""
    def last(*flags):
        v = None
        for i, a in enumerate(stock_args[:-1]):
            if a in flags:
                v = stock_args[i + 1]
        return v
    seeds_arg = last("--seeds", "-s")
    n_sample = int(last("--sample", "-e") or 5)
    out = {}
    for j in jobs or []:
        name = j.get("name")
        n_seeds = len([s for s in str(seeds_arg).split(",") if s.strip()]) if seeds_arg else max(1, len(j.get("modelSeeds") or []))
        out[name] = n_seeds * n_sample
    return {"per_item": out, "total": sum(out.values()), "n_sample": n_sample}


NONFINITE = "nonfinite_output"                          # the token of the finiteness gate: `PRED NONFINITE refused: exit 3 nonfinite_output=<item>/seed_<S>[,…]` (cli.py; SERVE likewise)
SUMMARY_NUMBERS = ("plddt", "ptm", "iptm", "ranking_score")   # upstream's summary_confidence keys read as numbers (null or NaN there = non-finite)
_CIF_NAN = re.compile(rb"(?<![A-Za-z0-9_.])[-+]?nan(?![A-Za-z0-9_])")   # a bare lowercase `nan` token (a float column; CCD / atom names are upper case)


def _nonfinite(v) -> bool:
    return isinstance(v, float) and (math.isnan(v) or math.isinf(v))


def _walk_numbers(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk_numbers(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_numbers(v)
    else:
        yield o


def nonfinite_cli(idx: dict) -> list:
    """The finiteness census of a CLI-route index: `<item>/seed_<S>` for every (item, seed) whose `*_summary_confidence_sample_<k>.json`
    holds a non-finite number (json `NaN` / `Infinity`, or null at plddt / ptm / iptm / ranking_score) or whose `*_sample_<k>.cif` carries a
    `nan` coordinate. A prediction that wrote every file but NaN numbers is never complete: `pred` refuses it with its own sentence
    (`PRED NONFINITE refused: exit 3 nonfinite_output=<item>/seed_<S>[,…]`) and the manifest's `nonfinite_outputs`; it is a census gate, not a
    lever's fallback. The rank subdirectories of an `--n_gpu` run (`.rowpair/`) are skipped."""
    root = idx.get("out_dir") or ""
    hits = set()
    for row in idx.get("files") or []:
        path = row.get("path") or ""
        rel = os.path.relpath(path, root) if root else path
        if rel.startswith(".rowpair/") or "/.rowpair/" in rel:
            continue
        m = re.match(r"([^/]+)/(seed_[^/]+)/predictions/[^/]*$", rel)
        if not m:
            continue
        unit = f"{m.group(1)}/{m.group(2)}"
        if unit in hits:
            continue
        if re.search(r"_summary_confidence_sample_\d+\.json$", rel):
            try:
                d = json.load(open(path))                  # Python's json reads NaN / Infinity tokens as floats
            except (OSError, ValueError):
                hits.add(unit)                             # an unreadable summary is not a finite one
                continue
            if any(_nonfinite(v) for v in _walk_numbers(d)) or any(k in d and not isinstance(d[k], (int, float)) for k in SUMMARY_NUMBERS):
                hits.add(unit)
        elif re.search(r"_sample_\d+\.cif$", rel):
            with open(path, "rb") as fh:
                if _CIF_NAN.search(fh.read()):
                    hits.add(unit)
    return sorted(hits)


def found_structures(idx: dict) -> dict:
    """Structure files per item in an index of the CLI route (`<name>/seed_<S>/predictions/<name>[_seed_<S>]_sample_<k>.cif`); the rank
    subdirectories of an `--n_gpu` run (`.rowpair/`) are not this process's outputs."""
    per = {}
    root = idx.get("out_dir") or ""
    for row in idx.get("files") or []:
        p = os.path.relpath(row.get("path") or "", root) if root else (row.get("path") or "")
        if p.startswith(".rowpair/") or "/.rowpair/" in p:
            continue
        m = re.match(r"([^/]+)/seed_[^/]+/predictions/.*_sample_\d+\.cif$", p)
        if m:
            per[m.group(1)] = per.get(m.group(1), 0) + 1
    return {"per_item": per, "total": sum(per.values())}
