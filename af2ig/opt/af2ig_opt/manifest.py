"""The driver's output layout under <out> as the package lays it out (cli.driver_args) and the reader of its items.

``<out>/pdbs/<tag>_af2pred.pdb``, the score file ``<out>/out.sc`` (`SCORE:` header + one row per design, description = the tag) and
``<out>/check.point`` are the driver's own outputs and the only files a run leaves; ``read_items`` joins the score rows with the PDBs
(the EXIT line's ``items=<ok>/<inputs>``).
"""
from __future__ import annotations

import os

from opt_core.gates import sha256_file

SCORE_FILE, PDB_DIR, CHECKPOINT = "out.sc", "pdbs", "check.point"       # the driver's own output names as the package lays them out (cli.driver_args)


def read_items(out_dir: str):
    """The driver's score file (out.sc: `SCORE:` lines, the header's last column `description` = the tag) joined with the output PDB per tag."""
    sc = os.path.join(out_dir, SCORE_FILE)
    if not os.path.isfile(sc):
        return None
    header, items = None, []
    with open(sc, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts or parts[0] != "SCORE:":
                continue
            if parts[-1] == "description":
                header = parts[1:-1]; continue
            if header is None:
                continue
            tag, vals = parts[-1], parts[1:-1]
            scores = dict(zip(header, vals))
            pdb = os.path.join(out_dir, PDB_DIR, tag + ".pdb")
            sha = sha256_file(pdb) if os.path.isfile(pdb) else None
            items.append({"tag": tag, "scores": {k: v for k, v in scores.items() if k != "time"}, "time_s": scores.get("time"), "pdb": os.path.relpath(pdb, out_dir),
                          "pdb_sha256": sha, "ok": sha is not None})
    return items
