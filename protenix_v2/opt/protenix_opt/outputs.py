"""The output census of a ``pred`` run: what the stock CLI was asked for against what it wrote (the exit rule: outputs short ->
``incomplete``, EXIT_FAIL; on both routes, printed as the OUTPUTS line).

The request is the resolved stock parameters (``stock_params``: ``input``, ``seeds``, ``sample``, ``use_seeds_in_json``) — every entry
of the input (a JSON file, a list of entries with ``name``; or a directory of such files) × every seed × every sample. The stock
dumper (stock ``runner/dumper.py``) writes each prediction as ``<name>/seed_<seed>/predictions/<name>_sample_<k>.cif`` with
``<name>_summary_confidence_sample_<k>.json`` beside it, under the output directory (or a dataset directory inside it); the census
looks for exactly those files. ``status`` is ``complete`` (every expected file present), ``incomplete`` (fewer: the missing names
listed) or ``unknown`` (the expectation cannot be computed: the named problem) — only ``complete`` is exit 0.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

CIF = "{name}_sample_{k}.cif"
SUMMARY = "{name}_summary_confidence_sample_{k}.json"
SEED_DIR = "seed_{seed}"
PRED_SUBDIR = "predictions"


def entries(input_path: Optional[str]) -> Tuple[List[str], List[str]]:
    """The entry names of a stock input (a JSON list of entries with ``name``, or a directory of such files, in name order); every
    entry that cannot be named is a problem."""
    names: List[str] = []
    problems: List[str] = []
    if not input_path:
        return names, ["no --input in the stock parameters"]
    if os.path.isdir(input_path):
        files = sorted(os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".json"))
        if not files:
            problems.append(f"{input_path}: a directory without .json files")
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        return names, [f"{input_path}: input not found"]
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as e:
            problems.append(f"{f}: not readable as JSON ({e!r})")
            continue
        if not isinstance(data, list):
            problems.append(f"{f}: not a list of entries")
            continue
        for i, e in enumerate(data):
            name = e.get("name") if isinstance(e, dict) else None
            if not name:
                problems.append(f"{f}: entry {i} without a name")
            else:
                names.append(str(name))
    return names, problems


def seeds_of(params: dict) -> Tuple[List[str], List[str]]:
    raw = str(params.get("seeds") if params.get("seeds") is not None else "101")
    seeds = [s.strip() for s in raw.split(",") if s.strip()]
    return seeds, ([] if seeds else [f"--seeds {raw!r}: no seed"])


def expected(params: dict) -> dict:
    """The files the request asks for, relative to the output directory: ``{"entries", "seeds", "samples", "files", "problems"}``."""
    names, problems = entries(params.get("input"))
    seeds, sp = seeds_of(params)
    problems += sp
    samples = params.get("sample")
    try:
        samples = int(samples) if samples is not None else 5
    except (TypeError, ValueError):
        problems.append(f"--sample {samples!r}: not an integer"); samples = 0
    files: List[str] = []
    for n in names:
        for s in seeds:
            for k in range(samples):
                d = os.path.join(n, SEED_DIR.format(seed=s), PRED_SUBDIR)
                files.append(os.path.join(d, CIF.format(name=n, k=k)))
                files.append(os.path.join(d, SUMMARY.format(name=n, k=k)))
    return {"entries": names, "seeds": seeds, "samples": samples, "files": files, "problems": problems}


def _present(out_dir: str, rel: str) -> bool:
    if os.path.isfile(os.path.join(out_dir, rel)):
        return True
    try:
        subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    except OSError:
        return False
    return any(os.path.isfile(os.path.join(out_dir, d, rel)) for d in subdirs)      # a dataset directory between out_dir and the entry


def census(out_dir: Optional[str], params: Optional[dict]) -> dict:
    """What was written against what was asked: ``{"status", "expected", "found", "missing", "entries", "seeds", "samples",
    "problems"}``; ``expected``/``found`` count files (a CIF and its summary JSON per sample)."""
    exp = expected(params or {})
    out = {"status": "unknown", "expected": len(exp["files"]), "found": 0, "missing": [], "entries": exp["entries"], "seeds": exp["seeds"],
           "samples": exp["samples"], "problems": list(exp["problems"])}
    if not out_dir:
        out["problems"].append("no --out_dir in the stock parameters")
    if out["problems"]:
        return out
    if not os.path.isdir(out_dir):
        out["missing"] = list(exp["files"])
        out["status"] = "incomplete"
        out["problems"].append(f"{out_dir}: output directory absent")
        return out
    missing = [rel for rel in exp["files"] if not _present(out_dir, rel)]
    out.update(found=len(exp["files"]) - len(missing), missing=missing, status="complete" if not missing else "incomplete")
    return out


def line(cen: dict, prefix: str) -> str:
    """One line: ``OUTPUTS complete 10/10`` or ``OUTPUTS incomplete 8/10 missing=…`` or ``OUTPUTS unknown: <problems>``."""
    if cen["status"] == "unknown":
        return f"{prefix} OUTPUTS unknown (completeness not provable): " + "; ".join(cen["problems"])
    head = f"{prefix} OUTPUTS {cen['status']} {cen['found']}/{cen['expected']} (entries={len(cen['entries'])} seeds={len(cen['seeds'])} samples={cen['samples']})"
    if cen["status"] == "incomplete":
        head += f" missing={cen['missing'][:10]}" + (f" (+{len(cen['missing']) - 10} more)" if len(cen["missing"]) > 10 else "")
    if cen.get("failures"):
        head += " failed=" + ",".join(f"{f['item']}:{f['reason_class']}" for f in cen["failures"][:10]) + (f" (+{len(cen['failures']) - 10} more)" if len(cen["failures"]) > 10 else "")
    return head
