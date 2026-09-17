"""``opt_manifest.json`` — the record ``design`` writes into the output directory through the core's document (``opt_core.manifest``: schema,
package, mode, route, activation, command, stack + core block, pass census, exit) with this kit's block under ``kit``: the route word, the
levers applied, the ``device`` the kit child's STARTUP record names (``cuda`` | ``cpu`` | ``none``), the sampling values, ``seed`` and
``batch`` (value, applied), ``det`` (requested, never applied), the inputs, the outputs, the weights staging facts and cache word, the children (argv, stamps, status — one kit child under ``fast``, one proven stock child
per structure under ``off``), the environment names removed for a stock child, its proof (``stock_env_proof.json`` read back), the script
``off`` ran, and ``lines`` — the evidence records read back from ``<out>/timing.jsonl``.
"""
from __future__ import annotations

import json
import os
from typing import List, Mapping, Optional

from opt_core import manifest as _core

from . import __version__, stack

SCHEMA = "esm_if1_opt.manifest/4"
FILENAME = _core.FILENAME                    # opt_manifest.json
TIMING = "timing.jsonl"


def read_records(path: str) -> List[dict]:
    """The evidence records of ``timing.jsonl``; a line that does not parse (a child killed mid-append) becomes a named
    ``{"kind": "MALFORMED", "line": n}`` record — counted, never dropped silently."""
    if not os.path.isfile(path):
        return []
    out: List[dict] = []
    with open(path, encoding="utf-8") as fh:
        for n, ln in enumerate(fh, 1):
            if not ln.strip():
                continue
            try:
                out.append(json.loads(ln))
            except ValueError:
                out.append({"kind": "MALFORMED", "line": n})
    return out


def read_proof(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def proof_ok(path: str) -> bool:
    p = read_proof(path)
    return bool(p and p.get("ok"))


def device_of(records: List[dict]) -> str:
    """The device the kit child's pass ran on — its STARTUP record's ``device`` (``cuda`` | ``cpu``); ``none`` when no STARTUP record was
    written (the child ended before the model load, or the stock route: upstream's script prints no STARTUP)."""
    return str(next((r.get("device") for r in records if r.get("kind") == "STARTUP" and r.get("device")), "none"))


def stack_block(records: List[dict]) -> dict:
    """The stack facts: the GPU probe of this process plus the torch / CUDA versions the pass's KERNELS record reported."""
    kern = next((r for r in records if r.get("kind") == "KERNELS"), {})
    return _core.stack_block(gpu=stack.gpu(), torch_version=kern.get("torch"), cuda=kern.get("cuda"), stack_key=stack.stack_key())


def write_run(out_dir: str, *, mode: str, route: str, report_: Mapping, command: List[str], kit: Mapping, census: Mapping, exit_: Mapping,
              wall_s: float, records: Optional[List[dict]] = None) -> str:
    """Assemble and write ``<out_dir>/opt_manifest.json``; returns its path. ``records``: the pass's evidence records when the caller has read
    them (``read_records``), else they are read here."""
    recs = read_records(os.path.join(out_dir, TIMING)) if records is None else list(records)
    kit_block = dict(kit)
    kit_block.setdefault("lines", recs)
    doc = _core.build(package="esm_if1_opt", package_version=__version__, schema=SCHEMA, mode=mode, report=report_, route=route,
                      command=list(command), stack=stack_block(recs), pass_=dict(census, wall_s=round(float(wall_s), 3)), exit_=dict(exit_), kit=kit_block)
    return _core.write(os.path.join(out_dir, FILENAME), doc)


def line(path: str) -> str:
    """``[esm_if1-opt] manifest: <path>`` (the core's line)."""
    from . import TAG
    return _core.line(TAG, path)
