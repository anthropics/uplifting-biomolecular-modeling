"""Content equality of two output trees: file digests, the per-run lines a writer stamps, the per-item comparison.

Contract. :func:`hash_outputs` digests every file under an output directory (``{relative path: sha256}``; the run's ``opt_manifest.json``
and the caller's work directories left out); :func:`content_sha256` is a file's digest with the lines a writer stamps per run dropped
(``drop_line_prefixes``: file suffix → the line prefixes, the caller's — every other file byte for byte); :func:`compare` is the census of two
digest maps over the union of paths (identical, total, differing, one-sided: total accounting); :func:`items_content` groups an output
directory's digests by item (the first path component, the launcher's staging prefix ``NNN_`` stripped from every component) and
:func:`compare_items` compares two such groupings item by item. Nothing here knows an engine's file names: suffixes, prefixes and the
excluded names are arguments.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .gates import sha256_file
from .manifest import FILENAME as MANIFEST_FILENAME

STAGED_STEM_RE = re.compile(r"^\d{3}_")                           # the launcher stages inputs as <i:03d>_<name>.json; the stem prefixes every output path component


def hash_outputs(out_dir: str, *, exclude_names: Iterable[str] = (MANIFEST_FILENAME,), exclude_dirs: Iterable[str] = ()) -> Dict[str, str]:
    """``{relative path: sha256}`` of every file under ``out_dir`` but the ``exclude_names`` (base names) and everything under the
    ``exclude_dirs`` (paths relative to ``out_dir``)."""
    names, dirs = set(exclude_names), [d.rstrip("/").replace("/", os.sep) for d in exclude_dirs]
    out: Dict[str, str] = {}
    for d, dns, fns in os.walk(out_dir):
        rel_d = os.path.relpath(d, out_dir)
        if any(rel_d == x or rel_d.startswith(x + os.sep) for x in dirs):
            dns[:] = []
            continue
        for f in sorted(fns):
            if f in names:
                continue
            p = os.path.join(d, f)
            out[os.path.relpath(p, out_dir).replace(os.sep, "/")] = sha256_file(p)
    return out


def content_sha256(path: str, drop_line_prefixes: Optional[Mapping[str, Iterable[str]]] = None) -> str:
    """sha256 of the file's content; for a file whose name ends with a key of ``drop_line_prefixes`` the lines starting with any of that
    key's prefixes are dropped first (a writer's per-run timestamp line) — every other file byte for byte."""
    with open(path, "rb") as fh:
        data = fh.read()
    for suffix, prefixes in (drop_line_prefixes or {}).items():
        if path.endswith(suffix):
            pre = tuple(p.encode() if isinstance(p, str) else p for p in prefixes)
            data = b"\n".join(ln for ln in data.split(b"\n") if not ln.startswith(pre))
    return hashlib.sha256(data).hexdigest()


def compare(a: Mapping[str, str], b: Mapping[str, str]) -> Tuple[int, int, List[str], List[str]]:
    """``(identical, total, differing, one_sided)`` over the union of the two digest maps' paths."""
    keys = sorted(set(a) | set(b))
    same = [k for k in keys if k in a and k in b and a[k] == b[k]]
    diff = [k for k in keys if k in a and k in b and a[k] != b[k]]
    only = [k for k in keys if (k in a) != (k in b)]
    return len(same), len(keys), diff, only


def items_content(out_dir: str, rels: Iterable[str], *, drop_line_prefixes: Optional[Mapping[str, Iterable[str]]] = None,
                  stem_re=STAGED_STEM_RE) -> Dict[str, Dict[str, str]]:
    """``{item: {path under the item: content sha256}}`` over the relative paths ``rels`` of ``out_dir``: the item is the first path component,
    ``stem_re`` (the staging prefix) is stripped from every component, the digest is :func:`content_sha256`."""
    out: Dict[str, Dict[str, str]] = {}
    for rel in rels:
        parts = [stem_re.sub("", p) for p in rel.split("/")]
        out.setdefault(parts[0], {})["/".join(parts[1:])] = content_sha256(os.path.join(out_dir, rel), drop_line_prefixes)
    return out


def compare_items(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]]) -> Dict[str, Tuple[int, int, List[str]]]:
    """Per item name over the union: ``(content-equal files, total files, differing + one-sided paths)``."""
    res: Dict[str, Tuple[int, int, List[str]]] = {}
    for name in sorted(set(a) | set(b)):
        same, total, diff, only = compare(a.get(name, {}), b.get(name, {}))
        res[name] = (same, total, diff + only)
    return res
