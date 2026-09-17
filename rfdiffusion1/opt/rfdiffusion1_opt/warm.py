"""``warm`` — the cold-start work of a mode done once, before the designs that count: the first design of each shape of the kit's
bundled example targets through ``design`` (TorchScript specialisation, graph capture — the first-position class of every shape, written
to a scratch directory that is not a result).

``warm --mode exact [KEY=VALUE ...] --out_dir D``: the shapes are the base kit's public cases (opt/forward/fast_inference/tests/cases_public.json:
``[{name, pdb, contigs, hotspots?, num_designs}]`` on the two bundled targets under opt/forward/fast_inference/inputs_public/) at one design
each, loaded here (load_cases) and handed to design.run as its rows; typed KEY=VALUE settings ride with them (a typed target key is a usage
error: the targets are the bundled ones). The design pass's own status and exit code are carried unchanged (the exit rule of report.verdict:
a partial warm-up exits 3 with its NOT ACTIVE line); its manifest lies under ``<out_dir>/first_designs/``.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

from . import design as _design, report as _report, stack, upstream_args
from .modes import TARGET_KEYS, resolve
from .report import EXIT_USAGE
from .registry import KIT_BASE

PUBLIC_CASES = (KIT_BASE, "tests/cases_public.json")                                # the base kit's example targets: warm's shapes
INPUT_DIRS = ((KIT_BASE, "inputs_public"),)                                            # where their input structures are bundled (the cases file names them by a path relative to itself; a path that does not resolve is found here by basename)
CASE_KEYS = ("name", "pdb", "contigs", "num_designs")                                  # a bundled case's required keys; hotspots optional (upstream's ppi.hotspot_res default: null)


def _input_dirs() -> List[str]:
    return [os.path.join(stack.kit_dir(k), *rel.split("/")) for k, rel in INPUT_DIRS]


def resolve_pdb(pdb: str, cases_dir: str, input_dirs: Optional[List[str]] = None) -> str:
    """The input PDB path this machine can read: as given when it exists; missing absolute paths by basename in the kit's bundled input
    directory; relative paths against the cases file. Refuses by name otherwise."""
    dirs = _input_dirs() if input_dirs is None else input_dirs
    if os.path.isabs(pdb):
        if os.path.isfile(pdb):
            return pdb
        for d in dirs:
            cand = os.path.join(d, os.path.basename(pdb))
            if os.path.isfile(cand):
                return cand
        raise _design.DesignError(f"input pdb not found: {pdb} (and no {os.path.basename(pdb)} under {', '.join(dirs)})")
    cand = os.path.join(cases_dir, pdb)
    if os.path.isfile(cand):
        return os.path.abspath(cand)
    for d in dirs:
        c2 = os.path.join(d, os.path.basename(pdb))
        if os.path.isfile(c2):
            return c2
    raise _design.DesignError(f"input pdb not found: {pdb} (relative to {cases_dir}; not under {', '.join(dirs)})")


def load_cases(path: str, num_designs: Optional[int] = None, input_dirs: Optional[List[str]] = None) -> List[dict]:
    """The bundled cases file as design.run's rows, validated and re-pointed for this machine (a NEW list; the file is never edited):
    `{name, pdb, contigs, hotspots, num_designs[, startnum]}` per case, `num_designs` = the given count when set."""
    raw = json.load(open(path, encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise _design.DesignError(f"cases file {path}: expected a non-empty JSON list of cases")
    cases_dir = os.path.dirname(os.path.abspath(path))
    out, names = [], set()
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or any(k not in c for k in CASE_KEYS):
            raise _design.DesignError(f"case #{i}: expected the keys {CASE_KEYS} (got {sorted(c) if isinstance(c, dict) else type(c).__name__})")
        name = str(c["name"])
        if not _design.NAME_RE.match(name) or name in names:
            raise _design.DesignError(f"case #{i}: name {name!r} must be unique and match {_design.NAME_RE.pattern} (it becomes the output directory)")
        names.add(name)
        d = dict(c)
        d["pdb"] = resolve_pdb(str(c["pdb"]), cases_dir, input_dirs)
        d["hotspots"] = str(c["hotspots"]) if c.get("hotspots") not in (None, "") else _design.NULL
        d["num_designs"] = int(num_designs if num_designs is not None else c["num_designs"])
        if "startnum" in c:
            d["startnum"] = int(c["startnum"])
        if d["num_designs"] < 1:
            raise _design.DesignError(f"case {name}: num_designs must be >= 1")
        if c.get("extra"):                                                              # per-case hydra overrides are not part of a cases file (the resident driver composes none per case): typed on the command line instead
            raise _design.DesignError(f"case {name}: extra {c['extra']!r}: a cases file carries no per-case overrides; hydra KEY=VALUE settings are typed on the "
                                      "command line")
        d.pop("extra", None)
        out.append(d)
    return out


def public_cases(overrides=None, num_designs: Optional[int] = 1) -> List[dict]:
    """The kit's bundled example targets as design.run's rows at `num_designs` each (PUBLIC_CASES through load_cases). A typed TARGET key
    beside them is a usage error by name: the targets are the bundled ones, the typed keys are settings that ride with every row."""
    typed_target = [k for k in upstream_args.parse([str(o) for o in (overrides or [])]).typed() if k in TARGET_KEYS and k != "inference.model_directory_path"]
    if typed_target:
        raise _design.UsageError(f"{typed_target} typed for warm: the targets are the kit's bundled example cases ({PUBLIC_CASES[1]}) — type settings only, "
                                 "or run `design` with the whole target")
    kit, rel = PUBLIC_CASES
    return load_cases(os.path.join(stack.kit_dir(kit), *rel.split("/")), num_designs)


def run(mode: Optional[str], out_dir: str, overrides=None, timeout: Optional[float] = None) -> dict:
    out_dir = os.path.abspath(out_dir)
    try:
        cases = public_cases(overrides, num_designs=1)                                # warm's rows, before anything is resolved: a typed target key is a usage error (exit 2), an unreadable bundled file a refusal by name
    except _design.UsageError as e:
        sys.stderr.write(f"rfdiffusion1-opt warm: {e}\n")
        return {"activation": None, "first_designs": None, "status": "usage", "rc": EXIT_USAGE, "reason": str(e)}
    except _design.DesignError as e:
        _report.emit(_report.not_active_line(str(e)))
        return {"activation": None, "first_designs": None, "status": "refused", "reason": str(e)}
    _, refused, overrides = _design.request(mode, overrides)                          # a request the kit line cannot serve: refused by name (design's NOT ACTIVE line), exit 3, nothing warmed or created
    if refused:
        return {"activation": None, "first_designs": None, **refused[1], "rc": refused[0]}
    os.makedirs(out_dir, exist_ok=True)
    rep = stack.activate(mode, overrides, dry_run=True)
    res_out = {"activation": rep, "first_designs": None, "status": "refused"}
    if rep.get("reason") or rep.get("would_refuse"):
        res_out["reason"] = "; ".join(rep.get("would_refuse") or [rep.get("reason")])
        return res_out
    res = resolve(rep["mode"], rep.get("overrides"))
    rc, man = _design.run(res.mode, list(res.settings.stock_overrides), cases=cases, out_dir=os.path.join(out_dir, "first_designs"), tag="warm", timeout=timeout)
    res_out["first_designs"] = {"rc": rc, "status": man.get("status"), "out_dir": os.path.join(out_dir, "first_designs"), "n_pdb": (man.get("outputs") or {}).get("n_pdb")}
    res_out["status"] = man.get("status") or ("ok" if rc == 0 else "failed")             # the design pass's own word: ok / partial / incomplete / failed / refused
    res_out["rc"] = rc                                                                  # and its exit code (3 partial or refused, 1 failed or incomplete)
    res_out["partial"] = man.get("partial") or []
    _report.emit(f"{_report.PREFIX} WARM {res_out['status']} mode={res.mode} first_designs={res_out['first_designs']['n_pdb']} out={out_dir}")
    return res_out
