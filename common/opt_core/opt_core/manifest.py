"""``opt_manifest.json`` — the record a run writes beside its outputs: the activation report, the box, the pass, the exit.

Contract. :func:`build` assembles the shared block — ``schema``, ``package``, ``package_version``, ``mode``, ``route``, ``activation``
(the report minus ``excluded_report_keys``), ``command``, ``stack`` (:func:`stack_block`: the box as found plus the ``core`` block),
``pass_status`` / ``n_items`` / ``n_items_complete`` / ``items_complete`` / ``item_failed`` (from ``pass_``), ``exit`` (the verdict) —
and the kit's own fields under ``kit`` and, when the kit says so, at the top level (``top_level``: a field a consumer reads there today
keeps its place). :func:`write` is atomic (tmp + rename), sorted keys, one trailing newline; :func:`read` is the inverse.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from typing import Iterable, Mapping, Optional, Sequence

FILENAME = "opt_manifest.json"
PASS_FIELDS = ("pass_status", "n_items", "n_items_complete", "items_complete", "item_failed")


def core_block() -> dict:
    """``{"version", "package_dir"}`` of the imported core (gates.imported_core) — the run record's fact of what actually stood in;
    equality is the git commit the tree is checked out at, not a restated hash."""
    from . import gates
    core = gates.imported_core()
    return {"version": core["version"], "package_dir": core["package_dir"]}


def stack_block(*, gpu: Optional[Mapping] = None, torch_version: Optional[str] = None, cuda: Optional[str] = None,
                stack_key: Optional[str] = None, extra: Optional[Mapping] = None) -> dict:
    """The box as found: python, torch (metadata version unless given), cuda, triton, the GPU probe, the stack key, the core block.
    Nothing is imported: versions come from distribution metadata; the GPU dict is the kit's probe (gates.nvidia_smi_probe / torch_gpu_probe)."""
    from . import gates
    block = {
        "python": platform.python_version(),
        "torch": torch_version if torch_version is not None else gates.dist_version("torch"),
        "cuda": cuda,
        "triton": gates.dist_version("triton"),
        "gpu": dict(gpu) if gpu else None,
        "stack_key": stack_key,
        "core": core_block(),
    }
    if extra:
        block.update(dict(extra))
    return block


def build(*, package: str, package_version: str, schema: str, mode: str, report: Optional[Mapping], route: str,
          command: Optional[Sequence[str]] = None, stack: Optional[Mapping] = None, pass_: Optional[Mapping] = None,
          exit_: Optional[Mapping] = None, kit: Optional[Mapping] = None, top_level: Iterable[str] = (),
          excluded_report_keys: Iterable[str] = ("logged",)) -> dict:
    """The manifest document (module contract). ``top_level`` names keys of ``kit`` copied to the top level as well."""
    rep = {k: v for k, v in dict(report or {}).items() if k not in set(excluded_report_keys)}
    doc = {
        "schema": schema,
        "package": package,
        "package_version": package_version,
        "mode": mode,
        "route": route,
        "activation": rep,
        "command": list(command) if command is not None else None,
        "stack": dict(stack) if stack is not None else None,
        "exit": dict(exit_) if exit_ is not None else None,
        "kit": dict(kit) if kit is not None else {},
    }
    p = dict(pass_ or {})
    for k in PASS_FIELDS:
        doc[k] = p.get(k)
    for k in top_level:
        if k in doc["kit"]:
            doc[k] = doc["kit"][k]
    return doc


def write(path: str, doc: Mapping) -> str:
    """Write ``doc`` atomically (a temp file beside it, then ``os.replace``), sorted keys, one trailing newline. Returns ``path``."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def line(tag: str, path: str) -> str:
    """``[tag] manifest: <path>`` — the one line a run prints when the manifest is written."""
    return f"[{tag}] manifest: {path}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m opt_core.manifest <path>`` prints the manifest's shared block as ``key=value`` lines (a reader for shell callers)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m opt_core.manifest <opt_manifest.json>", file=sys.stderr)
        return 2
    doc = read(argv[0])
    for k in ("schema", "package", "package_version", "mode", "route") + PASS_FIELDS:
        print(f"{k}={doc.get(k)}")
    ex = doc.get("exit") or {}
    print(f"exit_code={ex.get('exit_code')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
